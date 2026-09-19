#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <functional>

namespace ninfer {

// The events that enroll a SECOND device's stream in a capture that began on the first device's
// stream, and that order a replay of the result. Tensor-parallel decode issues work on both
// devices' streams and orders them
// against each other with cross-device event edges; those edges only become graph edges if both
// streams belong to the SAME capture. Two independent captures cannot be linked -- a
// cudaStreamWaitEvent across two live captures is rejected with cudaErrorStreamCaptureMerge -- so
// a tensor-parallel decode program is ONE graph holding both devices' nodes, not one graph per
// device.
//
//   record(fork) on the origin stream  ->  wait(fork) on the peer stream    (peer joins capture)
//   ...the whole two-device decode body...
//   record(join) on the peer stream    ->  wait(join) on the origin stream  (peer rejoins origin)
//
// The join is mandatory: cudaStreamEndCapture fails with cudaErrorStreamCaptureUnjoined if a
// forked stream is still outstanding. A third event serves gate_launch() below, which is about
// replay rather than capture. All three are created once (cudaEventCreate is not capturable) and
// one instance serves an unbounded number of sequential captures and launches.
class DecodeGraphPeerBridge {
public:
    DecodeGraphPeerBridge(int origin_device, int peer_device);
    ~DecodeGraphPeerBridge();

    DecodeGraphPeerBridge(const DecodeGraphPeerBridge&)            = delete;
    DecodeGraphPeerBridge& operator=(const DecodeGraphPeerBridge&) = delete;
    DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept;
    DecodeGraphPeerBridge& operator=(DecodeGraphPeerBridge&& other) noexcept;

    [[nodiscard]] int origin_device() const noexcept { return origin_device_; }
    [[nodiscard]] int peer_device() const noexcept { return peer_device_; }
    [[nodiscard]] cudaEvent_t fork_event() const noexcept { return fork_; }
    [[nodiscard]] cudaEvent_t join_event() const noexcept { return join_; }
    [[nodiscard]] bool live() const noexcept {
        return fork_ != nullptr && join_ != nullptr && gate_ != nullptr;
    }

    // REPLAY-side ordering, not capture-side. A dual-device graph is launched on the ORIGIN
    // device's stream; the graph's own edges then order its peer-device nodes after the graph
    // root, but they say nothing about work the caller already enqueued on the peer device's own
    // stream (at decode time: the mirrored KV page materialization). Eager execution gets that
    // ordering for free because it issues the peer's kernels on that same stream; a graph launch
    // does not, so the origin stream is explicitly ordered after the peer stream's outstanding
    // work before every launch, which transitively orders the whole graph after it.
    void gate_launch(cudaStream_t peer_stream, cudaStream_t origin_stream) const;

private:
    int origin_device_ = 0;
    int peer_device_   = 0;
    cudaEvent_t fork_  = nullptr;
    cudaEvent_t join_  = nullptr;
    cudaEvent_t gate_  = nullptr;
};

// The peer half of a dual-device capture: which stream to enroll, and the bridge that enrolls it.
struct DecodeGraphPeerCapture {
    const DecodeGraphPeerBridge* bridge = nullptr;
    cudaStream_t stream                 = nullptr;
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    // Single-device capture: `stream` is both the origin and the only stream captured.
    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Dual-device capture: `stream` is the origin (device 0) and `peer` names device 1's stream,
    // which is forked into the same capture for the duration of `body` and joined back before the
    // capture ends. A null `peer.bridge` is the single-device form above.
    void capture(cudaStream_t stream, const std::function<void()>& body,
                 const DecodeGraphPeerCapture& peer);
    [[nodiscard]] bool ready() const noexcept;
    // Node count of the captured graph, 0 when empty. Cross-device event edges are edges, not
    // nodes, so this counts real device work on BOTH devices.
    [[nodiscard]] std::size_t node_count() const;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
