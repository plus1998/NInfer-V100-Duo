// Opt-in dual-device CUDA Graph test for the tensor-parallel decode path.
//
// A tensor-parallel decode round issues work on BOTH devices' streams and orders them against each
// other with the collectives' cross-device events. Capturing that is only possible as ONE graph
// holding both devices' nodes -- rank 1's stream is forked into rank 0's capture and joined back
// before it ends -- because CUDA rejects an event wait that would link two separate live captures.
// This test is the durable evidence that the resulting graph is a faithful recording:
//
//   1. IDENTITY. Over 128 greedy decode steps the captured-and-replayed path must produce EXACTLY
//      the tokens the eager path produces. Greedy decoding is a hard equality test on every logit
//      argmax, so a graph that lost one device's contribution, replayed a stale address, or raced
//      the two streams diverges within a few tokens instead of degrading gracefully.
//   2. REPRODUCIBILITY. Two graph runs in the same engine must be bit-identical. The tp2-specific
//      hazard is a stream race, and a stable repeat is what rules it out.
//   3. BOTH DEVICES ARE IN THE GRAPH. One captured tp2 decode graph must hold more nodes than one
//      captured tp1 decode graph of the same model. tp2 runs the same per-layer schedule on each
//      of two devices and adds the collectives' cross-device copies, so its single graph is
//      roughly twice the size; a graph that had silently captured only rank 0's stream would come
//      out at or below the tp1 count. This is measured against the tp1 engine rather than a magic
//      constant, and it is granularity-free -- unlike the per-device byte figures, which
//      cudaMemGetInfo quantizes to 2 MiB and which are therefore reported but not asserted on.
//
// Legs 1 and 3 are complementary: 3 says the peer's nodes are IN the graph, 1 says the whole thing
// computes the right answer when replayed.
//
// Skipped (exit 77) unless NINFER_QWEN3_8_27B_WEIGHTS names an artifact AND at least two CUDA
// devices are visible, so it costs nothing in a default `ctest` run.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext   = 8192;
constexpr std::uint32_t kPrefillChunk = 1024;
// The floor for this test is 100 captured decode steps; 128 clears it.
constexpr std::uint32_t kOutputTokens = 128;
// Chosen so the execution frontier walks 460 -> 588 and therefore CROSSES this variant's graph
// profile boundary at 511. All of a family's ordinary profiles share one topology class and so one
// cudaGraphExec, which means the crossing makes the round call cudaGraphExecUpdate to swap the
// installed profile -- on a graph whose nodes live on two devices. A prompt that stayed inside one
// profile would never exercise that, and it is exactly the operation most likely to reject a
// cross-device topology.
constexpr std::size_t kPromptTokens = 460;

// The crossing is the point of those two constants, so it is pinned rather than left to a comment
// that a later edit could silently invalidate. `Variant::ordinary_graph_profiles` cuts the
// frontier range at {127, 511, 2047, 4095, ...} for this variant; the run must start below a
// boundary and end above it.
constexpr std::uint32_t kProfileBoundary = 511;
constexpr std::uint32_t kStartFrontier   = static_cast<std::uint32_t>(kPromptTokens);
constexpr std::uint32_t kEndFrontier     = kStartFrontier + kOutputTokens;
static_assert(kStartFrontier <= kProfileBoundary,
              "the prompt must start below the graph profile boundary");
static_assert(kEndFrontier > kProfileBoundary,
              "the generated tokens must carry the frontier past the graph profile boundary, so "
              "the round swaps the installed profile with cudaGraphExecUpdate on a cross-device "
              "graph; without the crossing this test proves strictly less");
static_assert(kEndFrontier < kMaxContext, "the run must fit the engine's context");
static_assert(kOutputTokens >= 100, "the required floor is 100 captured decode steps");

ninfer::EngineOptions engine_options(const char* artifact, int tp, bool graphs) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = kMaxContext;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk  = kPrefillChunk;
    options.kv_cache       = ninfer::KvCacheStorage::Int8Group64;
    options.tp             = tp;
    if (tp == 2) {
        options.devices = {0, 1};
    } else {
        options.device = 0;
    }
    options.use_cuda_graph = graphs;
    return options;
}

ninfer::EngineOptions tp2_options(const char* artifact, bool graphs) {
    return engine_options(artifact, 2, graphs);
}

std::vector<ninfer::TokenId> synthetic_prompt(std::size_t tokens) {
    std::vector<ninfer::TokenId> prompt;
    prompt.reserve(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + (index * 37) % 4096));
    }
    return prompt;
}

std::vector<ninfer::TokenId> generate_greedy(ninfer::Engine& engine,
                                             const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return engine.generate(engine.prepare_tokens(prompt), options).generated_token_ids;
}

std::size_t first_divergence(const std::vector<ninfer::TokenId>& a,
                             const std::vector<ninfer::TokenId>& b) {
    std::size_t index = 0;
    while (index < a.size() && index < b.size() && a[index] == b[index]) { ++index; }
    return index;
}

// Per-device graph accounting. `cuda_graph_bytes` is measured on each device with that device's
// own cudaMemGetInfo across the whole capture-and-instantiate window; it is REPORTED rather than
// asserted to be nonzero, because cudaMemGetInfo quantizes to 2 MiB and one decode graph for this
// model costs less than that once the modules are already resident. What is asserted is the part
// that is exact: the eager engine plans and consumes nothing, and neither device exceeds the
// planned per-device allowance. Leg 3 (node counts) is what proves the peer's work was captured.
int verify_graph_memory(const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
                        bool graphs) {
    if (load.tp != 2) {
        std::cerr << "tp2 load summary does not report two ranks\n";
        return 1;
    }
    const std::uint64_t rank0 = load.devices[0].cuda_graph_bytes;
    const std::uint64_t rank1 = load.devices[1].cuda_graph_bytes;
    if (!graphs) {
        if (rank0 != 0 || rank1 != 0 || memory.cuda_graph_allowance_bytes != 0 ||
            memory.cuda_graph_node_count != 0) {
            std::cerr << "the eager engine reserved or consumed graph resources (" << rank0 << " / "
                      << rank1 << " bytes, allowance " << memory.cuda_graph_allowance_bytes
                      << ", nodes " << memory.cuda_graph_node_count
                      << ") -- --no-cuda-graph is not actually disabling capture\n";
            return 1;
        }
        return 0;
    }
    if (memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "the graphs engine planned a zero graph allowance\n";
        return 1;
    }
    if (rank0 > memory.cuda_graph_allowance_bytes || rank1 > memory.cuda_graph_allowance_bytes) {
        std::cerr << "per-device graph residency (" << rank0 << " / " << rank1
                  << " B) exceeds the planned per-device allowance of "
                  << memory.cuda_graph_allowance_bytes << " B\n";
        return 1;
    }
    std::cout << "tp2 graph memory: rank0 " << rank0 << " B, rank1 " << rank1
              << " B, per-device allowance " << memory.cuda_graph_allowance_bytes << " B\n";
    return 0;
}

// Leg 3: one tp2 decode graph must hold substantially more nodes than one tp1 decode graph of the
// same model.
//
// The threshold is 1.5x, not "more than". tp2 runs the same per-layer schedule on EACH of two
// devices, so the honest expectation is ~2x plus the collectives' copies (measured 2.95x: 2x640
// for both devices, +512 for 128 all-reduces x 4 nodes, +4 for the gather, +96 for the tp2-only
// un-fused GDN norm). A bare "greater than" would still pass if only rank 0's stream had been
// captured and a handful of stray nodes made up the difference -- which is precisely the failure
// this leg exists to catch. 1.5x is comfortably below the real 2.95x and comfortably above
// anything a single-device capture could reach.
constexpr double kMinNodeRatio = 1.5;

int verify_both_devices_captured(std::size_t tp1_nodes, std::size_t tp2_nodes) {
    if (tp1_nodes == 0) {
        std::cerr << "the tp1 graphs engine captured no nodes -- the reference is missing\n";
        return 1;
    }
    const double ratio = static_cast<double>(tp2_nodes) / static_cast<double>(tp1_nodes);
    if (ratio < kMinNodeRatio) {
        std::cerr << "one tp2 decode graph holds " << tp2_nodes << " nodes against tp1's "
                  << tp1_nodes << " (" << ratio << "x, require at least " << kMinNodeRatio
                  << "x): a capture of both devices' schedules plus 128 cross-device collectives "
                     "cannot be this close to the one-device count, so the peer's half of the "
                     "schedule is missing from the graph\n";
        return 1;
    }
    std::cout << "tp2 decode graph nodes: " << tp2_nodes << " vs tp1's " << tp1_nodes << " ("
              << ratio << "x, require >= " << kMinNodeRatio
              << "x; both devices' work plus the collectives' copies)\n";
    return 0;
}

int exercise(const char* artifact) {
    const std::vector<ninfer::TokenId> prompt = synthetic_prompt(kPromptTokens);

    std::vector<ninfer::TokenId> eager;
    {
        ninfer::Engine engine(tp2_options(artifact, false));
        if (const int result =
                verify_graph_memory(engine.load_summary(), engine.memory_summary(), false);
            result != 0) {
            return result;
        }
        eager = generate_greedy(engine, prompt);
    }
    if (eager.size() != kOutputTokens) {
        std::cerr << "the eager tp2 engine did not generate " << kOutputTokens << " tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> captured;
    std::vector<ninfer::TokenId> replayed;
    std::size_t tp2_nodes = 0;
    {
        ninfer::Engine engine(tp2_options(artifact, true));
        if (const int result =
                verify_graph_memory(engine.load_summary(), engine.memory_summary(), true);
            result != 0) {
            return result;
        }
        tp2_nodes = engine.memory_summary().cuda_graph_node_count;
        captured  = generate_greedy(engine, prompt);
        replayed  = generate_greedy(engine, prompt);
    }
    std::size_t tp1_nodes = 0;
    {
        ninfer::Engine engine(engine_options(artifact, 1, true));
        tp1_nodes = engine.memory_summary().cuda_graph_node_count;
    }
    if (const int result = verify_both_devices_captured(tp1_nodes, tp2_nodes); result != 0) {
        return result;
    }
    if (captured.size() != kOutputTokens) {
        std::cerr << "the tp2 graphs engine did not generate " << kOutputTokens << " tokens\n";
        return 1;
    }
    if (captured != replayed) {
        std::cerr << "tp2 graph replay is not reproducible: the two runs first differ at token "
                  << first_divergence(captured, replayed) << " of " << kOutputTokens << '\n';
        return 1;
    }
    if (captured != eager) {
        std::cerr << "tp2 graph replay does not match the eager path: first divergence at token "
                  << first_divergence(captured, eager) << " of " << kOutputTokens << '\n';
        return 1;
    }
    std::cout << "tp2 graphs == eager over " << kOutputTokens
              << " greedy decode steps, and reproducible across two graph runs\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: dual-device graph capture test needs two CUDA devices\n";
        return 77;
    }
    if (exercise(artifact) != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
