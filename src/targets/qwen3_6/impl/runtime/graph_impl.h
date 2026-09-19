#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// A dual-device graph is launched ONCE, on rank 0's stream, and holds rank 1's nodes inside it.
// Launching it also makes rank 0's stream wait for the whole graph, so the caller's existing
// "synchronize the peer, then synchronize rank 0" is still sufficient to retire the round.
template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        if (state.execution.peer != nullptr) {
            const TpPeerCore& peer = *state.execution.peer;
            if (peer.graph_bridge == nullptr) {
                throw std::logic_error("tensor-parallel graph launch requires a peer bridge");
            }
            // See DecodeGraphPeerBridge::gate_launch: the eager path issues rank 1's kernels on
            // rank 1's own stream and is therefore automatically ordered after the round's
            // mirrored KV page materialization; a graph launched on rank 0's stream is not.
            peer.graph_bridge->gate_launch(peer.device->stream, state.execution.device.stream);
        }
        executable->launch(state.execution.device.stream);
    } else {
        body();
    }
}

// One capture site for every decode family, at tp1 and tp2 alike.
//
// At tp2 the body issues work on BOTH devices' streams and orders them with the collectives'
// cross-device events. Those events only become graph edges if both streams belong to the same
// capture, so rank 1's stream is forked into rank 0's capture for the duration of the body and
// joined back before it ends: one cudaGraph holding both devices' nodes, launched once on rank 0's
// stream. Two independent per-device captures are not an option -- CUDA rejects an event wait that
// would cross two live captures (cudaErrorStreamCaptureMerge).
//
// Both workspace arenas are reset first, for the same reason at both ranks: the body allocates its
// activations at deterministic arena offsets, and those addresses are baked into the graph.
template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    if (state.execution.peer == nullptr) {
        definition.capture(state.execution.device.stream, body);
        return;
    }
    const TpPeerCore& peer = *state.execution.peer;
    if (peer.graph_bridge == nullptr) {
        throw std::logic_error("tensor-parallel graph capture requires a peer capture bridge");
    }
    peer.work->reset();
    definition.capture(state.execution.device.stream, body,
                       DecodeGraphPeerCapture{peer.graph_bridge, peer.device->stream});
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
