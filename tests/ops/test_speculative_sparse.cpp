#include "ninfer/ops/speculative_round.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

void expect(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

float uniform(std::uint64_t seed, int position, int purpose) {
    std::uint64_t key = splitmix64(
        seed ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                0xD1B54A32D192ED03ull));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(purpose) << 21));
    return static_cast<float>(key >> 40) * (1.0f / 16777216.0f);
}

int run_case(int token_domain, int batch, int steps, bool partial) {
    const int physical_rows = token_domain == 248077 ? 248320 : token_domain;
    const int columns = steps + 1;
    std::vector<std::int32_t> targets(columns * batch, 0);
    std::vector<std::int32_t> drafts(steps * batch, 0);
    std::vector<std::int32_t> proposal_ids(16 * steps * batch);
    std::vector<float> proposal_q(16 * steps * batch, 0.0f);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                      f32_to_bf16(-20.0f));
    std::vector<ops::SamplingConfig> configs(batch);
    std::vector<DeviceBuffer> token_counts;
    token_counts.reserve(batch);
    std::vector<std::int32_t> extents(batch, steps);
    std::vector<std::int32_t> lengths(batch);
    std::vector<std::int32_t> expected_accepted(batch);
    std::vector<std::int32_t> expected_counts(batch);
    std::vector<std::int32_t> expected_tokens(columns * batch, 0);
    for (int row = 0; row < batch; ++row) {
        token_counts.emplace_back(static_cast<std::size_t>(token_domain) * sizeof(int));
        token_counts.back().fill(0);
        configs[row].token_counts = static_cast<int*>(token_counts.back().p);
        lengths[row] = 101 + row * 19;
        configs[row].temperature = row == batch - 1 ? 0.0f : 1.0f;
        configs[row].top_k = 1;
        for (int index = 0; index < steps; ++index) {
            drafts[row * steps + index] = 10 + row * 20 + index;
            const int base = (row * steps + index) * 16;
            for (int candidate = 0; candidate < 16; ++candidate) {
                proposal_ids[base + candidate] = 10 + row * 20 + candidate;
            }
            proposal_q[base + index] = 0.75f;
            proposal_q[base + (index + 1) % 16] = 0.25f;
        }
        const int accepting = row == 0 ? 0 : row == batch - 1 ? 1 : steps - 1;
        extents[row] = partial && row == 0 ? 0 : steps;
        expected_accepted[row] = std::min(accepting, extents[row]);
        expected_counts[row] = expected_accepted[row] + 1;
        for (int index = 0; index <= steps; ++index) {
            const int winner = index < accepting ? drafts[row * steps + index]
                                                   : 10 + row * 20 + steps + index;
            targets[row * columns + index] = winner;
            logits[(static_cast<std::size_t>(row) * columns + index) * physical_rows + winner] =
                f32_to_bf16(20.0f);
        }
        for (int index = 0; index < expected_accepted[row]; ++index) {
            expected_tokens[row * columns + index] = drafts[row * steps + index];
        }
        expected_tokens[row * columns + expected_accepted[row]] =
            targets[row * columns + expected_accepted[row]];
    }

    DeviceBuffer device_targets = to_device(targets);
    DeviceBuffer device_logits = to_device(logits);
    DeviceBuffer device_drafts = to_device(drafts);
    DeviceBuffer device_ids = to_device(proposal_ids);
    DeviceBuffer device_q = to_device(proposal_q);
    DeviceBuffer device_extents = to_device(extents);
    DeviceBuffer device_lengths = to_device(lengths);
    DeviceBuffer device_anchors = to_device(std::vector<int>(batch, -1));
    DeviceBuffer device_tokens = to_device(std::vector<int>(columns * batch, -1));
    DeviceBuffer device_counts = to_device(std::vector<int>(batch, -1));
    DeviceBuffer device_accepted = to_device(std::vector<int>(batch, -1));
    DeviceBuffer device_configs = to_device(configs);
    Tensor target(device_targets.p, DType::I32, {columns, batch});
    Tensor logit(device_logits.p, DType::BF16, {physical_rows, columns, batch});
    Tensor draft(device_drafts.p, DType::I32, {steps, batch});
    Tensor ids(device_ids.p, DType::I32, {16, steps, batch});
    Tensor q(device_q.p, DType::FP32, {16, steps, batch});
    Tensor extent(device_extents.p, DType::I32, {batch});
    Tensor length(device_lengths.p, DType::I32, {batch});
    Tensor anchor(device_anchors.p, DType::I32, {batch});
    Tensor tokens(device_tokens.p, DType::I32, {columns, batch});
    Tensor count(device_counts.p, DType::I32, {batch});
    Tensor accepted(device_accepted.p, DType::I32, {batch});
    const auto capacity = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
        token_domain, steps, steps, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::speculative_accept_sparse_drafts(
        target, logit, draft, ids, q, extent, length, anchor, tokens, count, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(device_configs.p), workspace,
        nullptr);
    cuda_synchronize();
    for (int row = 0; row < batch; ++row) { lengths[row] += expected_counts[row]; }
    std::vector<int> expected_anchors(batch);
    for (int row = 0; row < batch; ++row) {
        expected_anchors[row] = expected_tokens[row * columns + expected_accepted[row]];
    }
    int failures = 0;
    failures += verify_exact("sparse licensed tokens", from_device<int>(device_tokens, expected_tokens.size()),
                             expected_tokens);
    failures += verify_exact("sparse lengths", from_device<int>(device_lengths, batch), lengths);
    failures += verify_exact("sparse anchors", from_device<int>(device_anchors, batch), expected_anchors);
    failures += verify_exact("sparse accepted", from_device<int>(device_accepted, batch), expected_accepted);
    failures += verify_exact("sparse counts", from_device<int>(device_counts, batch), expected_counts);
    for (int row = 0; row < batch; ++row) {
        for (int index = 0; index < expected_counts[row]; ++index) {
            const int token = expected_tokens[row * columns + index];
            int actual = -1;
            cuda_check(cudaMemcpy(&actual,
                                  static_cast<const int*>(token_counts[row].p) + token,
                                  sizeof(actual), cudaMemcpyDeviceToHost),
                       "read sparse token count");
            const int expected = row == batch - 1 ? 0 : 1;
            if (actual != expected) {
                std::cerr << "sparse token count mismatch row=" << row << " token=" << token
                          << " got=" << actual << " expected=" << expected << '\n';
                ++failures;
            }
        }
    }
    expect(workspace.used() == 0 && workspace.peak_used() <= capacity,
           "sparse accept workspace exceeded its planned size");
    return failures;
}

int run_residual_case(int token_domain, bool should_accept) {
    constexpr int steps = 1;
    const int physical_rows = token_domain == 248077 ? 248320 : token_domain;
    std::uint64_t seed = 0;
    while (true) {
        const float draw = uniform(seed, 102, ops::kSamplePurposeSpeculativeAccept);
        if ((should_accept && draw > 0.50f && draw < 0.53f) ||
            (!should_accept && draw > 0.75f)) {
            break;
        }
        ++seed;
    }
    std::vector<std::uint16_t> logits(physical_rows * 2, f32_to_bf16(-20.0f));
    logits[10] = f32_to_bf16(0.0f);
    logits[11] = f32_to_bf16(0.0f);
    logits[physical_rows + 13] = f32_to_bf16(20.0f);
    DeviceBuffer device_logits = to_device(logits);
    DeviceBuffer device_targets = to_device<std::int32_t>({10, 13});
    DeviceBuffer device_drafts = to_device<std::int32_t>({10});
    std::vector<std::int32_t> ids(16);
    for (int index = 0; index < 16; ++index) { ids[index] = 10 + index; }
    DeviceBuffer device_ids = to_device(ids);
    std::vector<float> probabilities(16, 0.0f);
    probabilities[0] = 0.9f;
    probabilities[1] = 0.1f;
    DeviceBuffer device_q = to_device(probabilities);
    DeviceBuffer device_extents = to_device<std::int32_t>({steps});
    DeviceBuffer device_lengths = to_device<std::int32_t>({101});
    DeviceBuffer device_anchors = to_device<std::int32_t>({-1});
    DeviceBuffer device_tokens = to_device<std::int32_t>({-1, -1});
    DeviceBuffer device_count = to_device<std::int32_t>({-1});
    DeviceBuffer device_accepted = to_device<std::int32_t>({-1});
    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k = 2;
    config.seed = seed;
    DeviceBuffer device_config = to_device(std::vector<ops::SamplingConfig>{config});
    Tensor target(device_targets.p, DType::I32, {2});
    Tensor logit(device_logits.p, DType::BF16, {physical_rows, 2});
    Tensor draft(device_drafts.p, DType::I32, {1});
    Tensor proposal_ids(device_ids.p, DType::I32, {16, 1, 1});
    Tensor proposal_q(device_q.p, DType::FP32, {16, 1, 1});
    Tensor extent(device_extents.p, DType::I32, {1});
    Tensor length(device_lengths.p, DType::I32, {1});
    Tensor anchor(device_anchors.p, DType::I32, {1});
    Tensor licensed(device_tokens.p, DType::I32, {2});
    Tensor count(device_count.p, DType::I32, {1});
    Tensor accepted(device_accepted.p, DType::I32, {1});
    const auto capacity = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
        token_domain, 1, 1, 1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::speculative_accept_sparse_drafts(
        target, logit, draft, proposal_ids, proposal_q, extent, length, anchor, licensed,
        count, accepted, token_domain,
        static_cast<const ops::SamplingConfig*>(device_config.p), workspace, nullptr);
    cuda_synchronize();
    int failures = 0;
    const std::vector<int> expected_tokens = should_accept ? std::vector<int>{10, 13}
                                                           : std::vector<int>{11, 0};
    const int expected_length = should_accept ? 103 : 102;
    const int expected_anchor = should_accept ? 13 : 11;
    failures += verify_exact("sparse residual token", from_device<int>(device_tokens, 2),
                             expected_tokens);
    failures += verify_exact("sparse residual length", from_device<int>(device_lengths, 1),
                             {expected_length});
    failures += verify_exact("sparse residual anchor", from_device<int>(device_anchors, 1),
                             {expected_anchor});
    failures += verify_exact("sparse residual accepted", from_device<int>(device_accepted, 1),
                             {should_accept ? 1 : 0});
    Tensor overlapping_q(device_tokens.p, DType::FP32, {16, 1, 1});
    try {
        ops::speculative_accept_sparse_drafts(
            target, logit, draft, proposal_ids, overlapping_q, extent, length, anchor,
            licensed, count, accepted, token_domain,
            static_cast<const ops::SamplingConfig*>(device_config.p), workspace, nullptr);
        throw std::runtime_error("sparse accept admitted output/proposal aliasing");
    } catch (const std::invalid_argument&) {
    }
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "sparse graph stream");
    const int initial_length = 101;
    device_lengths.copy_from_host(&initial_length, sizeof(initial_length));
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "sparse graph capture begin");
    ops::speculative_accept_sparse_drafts(
        target, logit, draft, proposal_ids, proposal_q, extent, length, anchor, licensed,
        count, accepted, token_domain,
        static_cast<const ops::SamplingConfig*>(device_config.p), workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "sparse graph capture end");
    cuda_check(cudaGraphInstantiate(&executable, graph, 0), "sparse graph instantiate");
    cuda_check(cudaGraphLaunch(executable, stream), "sparse graph replay");
    cuda_synchronize(stream);
    failures += verify_exact("sparse graph residual token", from_device<int>(device_tokens, 2),
                             expected_tokens);
    failures += verify_exact("sparse graph residual length", from_device<int>(device_lengths, 1),
                             {expected_length});
    cuda_check(cudaGraphExecDestroy(executable), "sparse graph destroy exec");
    cuda_check(cudaGraphDestroy(graph), "sparse graph destroy");
    cuda_check(cudaStreamDestroy(stream), "sparse graph destroy stream");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        failures += run_case(64, 3, 3, true);
        failures += run_case(248077, 3, 3, false);
        failures += run_case(248077, 8, 15, true);
        failures += run_residual_case(64, false);
        failures += run_residual_case(64, true);
        failures += run_residual_case(248077, false);
        failures += run_residual_case(248077, true);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " speculative sparse\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "speculative sparse: " << error.what() << '\n';
        return 1;
    }
}
