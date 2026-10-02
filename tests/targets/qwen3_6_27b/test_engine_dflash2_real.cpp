#include "ninfer/engine.h"
#include "../qwen3_6/speculative_page_boundary.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void valid(const ninfer::GenerationResult& result, std::size_t outputs) {
    require(result.generated_token_ids.size() == outputs &&
                result.finish_reason == ninfer::FinishReason::OutputLimit,
            "DFlash2 did not honor the requested output budget");
    require(
        result.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
            (outputs == 1 || result.speculative.rounds + result.speculative.fallback_steps != 0),
        "generation bypassed DFlash2");
}

ninfer::PromptInput media_prompt(ninfer::MediaKind kind) {
    const std::string header = "P6\n64 64\n255\n";
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = kind;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "pattern.ppm";
    media.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        media.media.bytes.push_back(i & 255);
        media.media.bytes.push_back((i * 3) & 255);
        media.media.bytes.push_back((i * 7) & 255);
    }
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back({.kind  = ninfer::MessagePartKind::Text,
                          .text  = "Describe the pattern briefly.",
                          .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}
} // namespace

// Optional K, Graph, optimized-head, B and KV codec arguments select representative integration
// routes without multiplying test binaries. The artifact explicitly selects the storage profile.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS is not set\n";
        return 77;
    }
    try {
        const auto k         = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 15U;
        const bool graph     = argc > 2 ? std::stoi(argv[2]) != 0 : true;
        const bool optimized = argc > 3 ? std::stoi(argv[3]) != 0 : true;
        const auto batch     = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 8U;
        const bool measure_speed = argc > 8 && std::stoi(argv[8]) != 0;
        const bool dflash_only = argc > 9 && std::stoi(argv[9]) != 0;
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.tp              = 2;
        options.devices         = {0, 1};
        options.max_context     = 2304;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * batch);
        options.prefill_chunk   = 256;
        options.max_concurrency = batch;
        options.use_cuda_graph                   = graph;
        options.enable_vision                    = argc > 6 && std::stoi(argv[6]) != 0;
        options.kv_cache                         = argc > 5 && std::string(argv[5]) == "int8"
                                                       ? ninfer::KvCacheStorage::Int8Group64
                                                       : ninfer::KvCacheStorage::BFloat16;
        std::vector<ninfer::TokenId> prompt, reference, penalty_reference;
        std::optional<double> mtp_decode_rate;
        std::optional<double> ordinary_decode_rate;
        auto penalty                                 = request(24);
        penalty.execution.sampling.presence_penalty  = 0.5F;
        penalty.execution.sampling.frequency_penalty = 0.25F;
        {
            auto ordinary_options            = options;
            ordinary_options.max_concurrency = 1;
            ordinary_options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304);
            ordinary_options.use_cuda_graph  = false;
            ordinary_options.enable_vision   = false;
            ninfer::Engine ordinary(ordinary_options);
            ninfer::PromptInput input;
            input.options.enable_thinking = false;
            ninfer::ChatMessage message;
            const char* profile_text = std::getenv("NINFER_DFLASH2_PROFILE_TEXT");
            message.parts.push_back({ninfer::MessagePartKind::Text,
                                     profile_text != nullptr && *profile_text != '\0'
                                         ? profile_text
                                         : "Count from one to twenty: one, two, three,", {}});
            input.messages.push_back(std::move(message));
            prompt = ordinary.prepare(std::move(input)).debug_token_ids();
            reference =
                ordinary.generate(ordinary.prepare_tokens(prompt), request(24)).generated_token_ids;
            penalty_reference =
                ordinary.generate(ordinary.prepare_tokens(prompt), penalty).generated_token_ids;
        }
        const auto decode_rate = [&](ninfer::Engine& current) {
            current.generate(current.prepare_tokens(prompt), request(16));
            double decode_seconds = 0.0;
            constexpr int repetitions = 3;
            constexpr std::uint32_t generated = 128;
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                const auto result = current.generate(current.prepare_tokens(prompt),
                                                     request(generated));
                require(result.generated_token_ids.size() == generated &&
                            result.finish_reason == ninfer::FinishReason::OutputLimit,
                        "speed probe did not exhaust its output budget");
                decode_seconds += result.timings.decode_seconds;
            }
            return static_cast<double>(repetitions * (generated - 1)) / decode_seconds;
        };
        if (measure_speed && batch == 1) {
            {
                ninfer::Engine ordinary(options);
                ordinary_decode_rate = decode_rate(ordinary);
            }
            auto mtp_options = options;
            mtp_options.enable_vision = false;
            mtp_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            mtp_options.speculative.draft_tokens = 3;
            mtp_options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            ninfer::Engine mtp(mtp_options);
            mtp_decode_rate = decode_rate(mtp);
        }
        if (const char* profile = std::getenv("NINFER_DFLASH2_PROFILE");
            profile != nullptr && std::string_view(profile) == "mtp") {
            auto mtp_options = options;
            mtp_options.enable_vision = false;
            mtp_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            mtp_options.speculative.draft_tokens = 3;
            mtp_options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            ninfer::Engine mtp(mtp_options);
            mtp.generate(mtp.prepare_tokens(prompt), request(16));
            require(cudaProfilerStart() == cudaSuccess, "cannot start CUDA profiler");
            const auto measured = mtp.generate(mtp.prepare_tokens(prompt), request(128));
            require(cudaProfilerStop() == cudaSuccess, "cannot stop CUDA profiler");
            require(measured.generated_token_ids.size() == 128,
                    "MTP profile did not exhaust its output budget");
            std::cout << "profile MTP3 graph=" << graph
                      << " decode_s=" << measured.timings.decode_seconds
                      << " rounds=" << measured.speculative.rounds
                      << " accepted=" << measured.speculative.accepted_tokens
                      << "/" << measured.speculative.drafted_tokens << " per_position=";
            for (const auto accepted : measured.speculative.accepted_per_position) {
                std::cout << accepted << ',';
            }
            std::cout << '\n';
            return 0;
        }
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = k;
        options.speculative.proposal_head =
            optimized ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        if (measure_speed && batch > 1) {
            constexpr int repetitions = 3;
            constexpr std::uint32_t generated = 128;
            const auto batch_probe = [&](ninfer::Engine& current,
                                         ninfer::SpeculativeBackend backend) {
                current.generate(current.prepare_tokens(prompt), request(16));
                double seconds = 0.0;
                std::uint64_t rounds = 0;
                std::uint64_t accepted = 0;
                std::uint64_t drafted = 0;
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    std::vector<ninfer::GenerationHandle> handles;
                    handles.reserve(batch);
                    const auto started = std::chrono::steady_clock::now();
                    for (unsigned row = 0; row < batch; ++row) {
                        handles.push_back(current.submit(current.prepare_tokens(prompt),
                                                         request(generated)));
                    }
                    for (auto& handle : handles) {
                        const auto result = handle.wait();
                        require(result.generated_token_ids.size() == generated &&
                                    result.finish_reason == ninfer::FinishReason::OutputLimit &&
                                    result.speculative.backend == backend,
                                "concurrent speed probe returned an incomplete request");
                        rounds += result.speculative.rounds;
                        accepted += result.speculative.accepted_tokens;
                        drafted += result.speculative.drafted_tokens;
                    }
                    seconds += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started).count();
                }
                return std::tuple{static_cast<double>(repetitions * batch * generated) / seconds,
                                  rounds, accepted, drafted};
            };
            auto mtp_options = options;
            mtp_options.enable_vision = false;
            mtp_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            mtp_options.speculative.draft_tokens = 3;
            mtp_options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
            std::optional<std::tuple<double, std::uint64_t, std::uint64_t, std::uint64_t>> mtp_result;
            if (!dflash_only) {
                ninfer::Engine mtp(mtp_options);
                mtp_result = batch_probe(mtp, ninfer::SpeculativeBackend::Mtp);
            }
            const auto [dflash_rate, dflash_rounds, dflash_accepted, dflash_drafted] = [&] {
                ninfer::Engine engine(options);
                return batch_probe(engine, ninfer::SpeculativeBackend::DFlash2);
            }();
            std::cout << "concurrent 3x128 outputs B=" << batch << " graph=" << graph
                      << " DFlash2-K=" << k << "=" << dflash_rate << " tok/s ("
                      << dflash_rounds << " rounds, "
                      << dflash_accepted << "/" << dflash_drafted << " accepted)\n";
            if (mtp_result) {
                const auto [mtp_rate, mtp_rounds, mtp_accepted, mtp_drafted] = *mtp_result;
                std::cout << "MTP3=" << mtp_rate << " tok/s (" << mtp_rounds << " rounds, "
                          << mtp_accepted << "/" << mtp_drafted << " accepted)\n";
            }
            return 0;
        }
        ninfer::Engine engine(options);
        if (std::getenv("NINFER_DFLASH2_PROFILE") != nullptr) {
            engine.generate(engine.prepare_tokens(prompt), request(16));
            require(cudaProfilerStart() == cudaSuccess, "cannot start CUDA profiler");
            const auto measured = engine.generate(engine.prepare_tokens(prompt), request(128));
            require(cudaProfilerStop() == cudaSuccess, "cannot stop CUDA profiler");
            valid(measured, 128);
            std::cout << "profile K=" << k << " graph=" << graph
                      << " decode_s=" << measured.timings.decode_seconds
                      << " rounds=" << measured.speculative.rounds
                      << " accepted=" << measured.speculative.accepted_tokens
                      << "/" << measured.speculative.drafted_tokens << " per_position=";
            for (const auto accepted : measured.speculative.accepted_per_position) {
                std::cout << accepted << ',';
            }
            std::cout << '\n';
            return 0;
        }
        ninfer::test::speculative_page_boundary(engine);
        const auto first = engine.generate(engine.prepare_tokens(prompt), request(24));
        valid(first, 24);
        require(first.generated_token_ids == reference,
                "DFlash2 greedy target result differs from ordinary decoding");
        require(first.speculative.accepted_tokens != 0, "real draft fixture accepted no proposal");
        const auto penalized = engine.generate(engine.prepare_tokens(prompt), penalty);
        valid(penalized, 24);
        require(penalized.generated_token_ids == penalty_reference,
                "DFlash2 committed token counts differ from ordinary decoding");

        // All rows share a known target prefix, while their budgets force P=0, partial and full W.
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned row = 0; row < batch; ++row) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(2 + row * 3)));
        }
        for (unsigned row = 0; row < batch; ++row) {
            const auto result = handles[row].wait();
            valid(result, 2 + row * 3);
            require(std::equal(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                               reference.begin()),
                    "compact DFlash2 batch changed a row's target result");
        }
        // Rejection RNG uses the same logical positions and seed for repeat requests.
        auto sampled                                 = request(16);
        sampled.execution.sampling.temperature       = 0.8F;
        sampled.execution.sampling.top_p             = 0.9F;
        sampled.execution.sampling.top_k             = 20;
        sampled.execution.sampling.presence_penalty  = 0.3F;
        sampled.execution.sampling.frequency_penalty = 0.2F;
        sampled.execution.sampling.seed              = 42;
        const auto sample1 = engine.generate(engine.prepare_tokens(prompt), sampled);
        const auto sample2 = engine.generate(engine.prepare_tokens(prompt), sampled);
        valid(sample1, 16);
        require(sample1.generated_token_ids == sample2.generated_token_ids,
                "same DFlash2 seed and inputs did not reproduce the conditional path");

        // Terminal flush and fork: reuse the consumed prefix, then compare with a fresh prefill.
        const auto retained = engine.generate(engine.prepare_tokens(prompt), request(12, true));
        auto continuation   = prompt;
        continuation.insert(continuation.end(), retained.generated_token_ids.begin(),
                            retained.generated_token_ids.end());
        continuation.push_back(198);
        const auto reused = engine.generate(engine.prepare_tokens(continuation), request(8, true));
        const auto fresh  = engine.generate(engine.prepare_tokens(continuation), request(8, false));
        require(reused.reused_prompt_tokens != 0 &&
                    reused.generated_token_ids == fresh.generated_token_ids,
                "DFlash2 prefix restore did not preserve target and local context state");

        if (k >= 7) {
            bool checked_partial = false;
            for (std::size_t i = 1; i < std::min<std::size_t>(k, reference.size()); ++i) {
                if (std::find(reference.begin(), reference.begin() + i, reference[i]) !=
                    reference.begin() + i) {
                    continue;
                }
                auto stopped_options = request(24, true);
                stopped_options.stop.token_ids.push_back(reference[i]);
                const auto stopped =
                    engine.generate(engine.prepare_tokens(prompt), stopped_options);
                const auto licensed = 1 + stopped.speculative.rounds +
                                      stopped.speculative.accepted_tokens +
                                      stopped.speculative.fallback_steps;
                if (stopped.generated_token_ids.size() >= licensed) { continue; }
                require(stopped.finish_reason == ninfer::FinishReason::StopToken,
                        "partial terminal did not stop at its token");
                auto follow = prompt;
                follow.insert(follow.end(), stopped.generated_token_ids.begin(),
                              stopped.generated_token_ids.end());
                follow.push_back(198);
                const auto reused_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, true));
                const auto fresh_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, false));
                require(reused_stop.reused_prompt_tokens != 0 &&
                            reused_stop.generated_token_ids == fresh_stop.generated_token_ids,
                        "partial terminal folded the wrong GDN/hidden/context prefix");
                checked_partial = true;
                break;
            }
            require(checked_partial, "fixture did not exercise a stop within a licensed block");
        }
        if (options.enable_vision) {
            for (const auto kind : {ninfer::MediaKind::Image, ninfer::MediaKind::Video}) {
                const auto image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                const auto reused_image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                valid(image, 8);
                require(image.prompt.has_media && image.timings.vision_seconds > 0 &&
                            reused_image.reused_prompt_tokens != 0 &&
                            image.generated_token_ids == reused_image.generated_token_ids,
                        "Vision DFlash2 capture/restore changed the result");
            }
        }
        const auto stats = engine.runtime_stats();
        if (k == 15) {
            // One oversized prefill replaces the ring, then decode appends across its wrap point.
            auto long_prompt = std::vector<ninfer::TokenId>(2100, 198);
            long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
            const auto long_run =
                engine.generate(engine.prepare_tokens(long_prompt), request(12, true));
            valid(long_run, 12);
            long_prompt.insert(long_prompt.end(), long_run.generated_token_ids.begin(),
                               long_run.generated_token_ids.end());
            const auto long_reuse =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, true));
            const auto long_fresh =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, false));
            require(long_reuse.reused_prompt_tokens > 2048 &&
                        long_reuse.generated_token_ids == long_fresh.generated_token_ids,
                    "DFlash2 ring wrap/retained prefix changed the result");
        }
        if (k == 15) {
            auto tail_prompt   = std::vector<ninfer::TokenId>(options.max_context - 4, 198);
            tail_prompt.back() = prompt.back();
            const auto tail    = engine.generate(engine.prepare_tokens(tail_prompt), request(9));
            require(tail.finish_reason == ninfer::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == 5,
                    "full proposal window escaped the target context capacity tail");
        }
        if (mtp_decode_rate) {
            const double dflash2_decode_rate = decode_rate(engine);
            std::cout << "decode-only 3x127 tokens, TP2, B=1, graph=" << graph
                      << " ordinary=" << *ordinary_decode_rate << " tok/s MTP3="
                      << *mtp_decode_rate << " tok/s DFlash2-K=" << k
                      << "=" << dflash2_decode_rate << " tok/s\n";
        }
        std::cout << "ok K=" << k << " B=" << batch << " graph=" << graph
                  << " optimized=" << optimized << " accepted=" << first.speculative.accepted_tokens
                  << "/" << first.speculative.drafted_tokens
                  << " decode_rounds=" << stats.decode_rounds << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
