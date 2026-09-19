// Implements: src/core/decode_graph.h
//
// THE MULTI-DEVICE CONSTRAINTS THAT ACTUALLY BIND, and what each one forces.
// Measured on CUDA 13.1 / driver 580.178.04 by tools/tp2/capture_probe.cu and
// tools/tp2/transport_probe.cu; re-run both on any new two-GPU topology.
//
//  1. TWO LIVE CAPTURES CANNOT BE LINKED. cudaStreamWaitEvent on an event recorded inside a
//     *different* live capture fails with cudaErrorStreamCaptureMerge. Since a tensor-parallel
//     decode program's only cross-device ordering IS such an event edge, "one graph per device"
//     is not expressible. Forces: ONE graph holding both devices' nodes, with the peer's stream
//     forked into the origin's capture (DecodeGraphPeerBridge).
//
//  2. MEMCPY NODES MUST NAME THEIR MEMORY BY UVA POINTER, not by device id. cudaMemcpyPeerAsync
//     -- the explicit (ptr, device, ptr, device) form -- is rejected inside a capture region with
//     cudaErrorStreamCaptureUnsupported. cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA
//     pointers expresses the same cross-device transfer and IS captured, because a UVA pointer
//     already resolves to its device and the node needs no separate context argument. Forces:
//     src/ops/common/allreduce.cu pulls with the UVA form (see its pull_peer()).
//
//  3. cudaGraphExecUpdate REQUIRES THE SAME TOPOLOGY, INCLUDING NODE DEVICE RESIDENCY. Swapping
//     an installed profile into an existing executable only works while every node keeps the
//     device and context it was captured on. That holds here because all profiles of a family
//     are captured from the same body against the same two DeviceContexts; it is why the
//     profile-boundary swap is exercised deliberately in tests/targets/qwen3_6_27b/
//     test_graph_tp2.cpp rather than left to chance.
//
//  4. NO DEVICE-SIDE LAUNCH. cudaGraphInstantiate is deliberately called with flags 0, NOT
//     cudaGraphInstantiateFlagDeviceLaunch: device-launchable graphs must be single-device, so
//     requesting that flag would reject exactly the cross-device graph this file exists to build.
//
//  5. EVENT RECORDS AND WAITS BECOME EDGES, NOT NODES. A captured graph therefore holds only real
//     device work; the collectives' four-event choreography costs zero nodes. This is what makes
//     the node count a usable measurement of "did both devices' work get captured".
#include "core/decode_graph.h"

#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

// Current-device save/restore. Enrolling and retiring the peer stream requires that device to be
// current for the record and this one for the wait, and the caller's device must survive.
class ScopedDevice {
public:
    ScopedDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~ScopedDevice() { log_cuda_error("cudaSetDevice(restore)", cudaSetDevice(previous_)); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

// Fork the peer stream into the capture that `stream` (the origin) has already begun.
void fork_peer(cudaStream_t stream, const DecodeGraphPeerCapture& peer) {
    const ScopedDevice scope;
    ScopedDevice::set(peer.bridge->origin_device());
    CUDA_CHECK(cudaEventRecord(peer.bridge->fork_event(), stream));
    ScopedDevice::set(peer.bridge->peer_device());
    CUDA_CHECK(cudaStreamWaitEvent(peer.stream, peer.bridge->fork_event(), 0));
}

// Join the peer stream back into the origin. Without this cudaStreamEndCapture reports
// cudaErrorStreamCaptureUnjoined and the whole capture is discarded.
void join_peer(cudaStream_t stream, const DecodeGraphPeerCapture& peer) {
    const ScopedDevice scope;
    ScopedDevice::set(peer.bridge->peer_device());
    CUDA_CHECK(cudaEventRecord(peer.bridge->join_event(), peer.stream));
    ScopedDevice::set(peer.bridge->origin_device());
    CUDA_CHECK(cudaStreamWaitEvent(stream, peer.bridge->join_event(), 0));
}

void discard_capture(cudaStream_t stream, const DecodeGraphPeerCapture* peer,
                     bool peer_forked) noexcept {
    // Best effort: rejoin the peer so the origin's EndCapture is well formed. If the capture was
    // already invalidated these calls fail harmlessly and EndCapture then returns a null graph and
    // clears BOTH streams' capture state, which is the outcome that matters. Hand-rolled rather
    // than ScopedDevice because this runs on an exception path and must not throw.
    int caller_device = 0;
    const bool restore =
        peer != nullptr && peer_forked && cudaGetDevice(&caller_device) == cudaSuccess;
    if (peer != nullptr && peer_forked) {
        log_cuda_error("cudaSetDevice(peer)", cudaSetDevice(peer->bridge->peer_device()));
        log_cuda_error("cudaEventRecord(join)",
                       cudaEventRecord(peer->bridge->join_event(), peer->stream));
        log_cuda_error("cudaSetDevice(origin)", cudaSetDevice(peer->bridge->origin_device()));
        log_cuda_error("cudaStreamWaitEvent(join)",
                       cudaStreamWaitEvent(stream, peer->bridge->join_event(), 0));
    }
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
    if (peer != nullptr && peer_forked) {
        // A stream left in capture mode would poison every later launch on it, so say so loudly
        // rather than failing mysteriously later.
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(peer->stream, &status) == cudaSuccess &&
            status != cudaStreamCaptureStatusNone) {
            std::fprintf(stderr,
                         "CUDA cleanup failed: peer stream is still capturing after a discarded "
                         "dual-device capture\n");
        }
    }
    if (restore) { log_cuda_error("cudaSetDevice(restore)", cudaSetDevice(caller_device)); }
}

} // namespace

DecodeGraphPeerBridge::DecodeGraphPeerBridge(int origin_device, int peer_device)
    : origin_device_(origin_device), peer_device_(peer_device) {
    if (origin_device == peer_device) {
        throw std::invalid_argument(
            "DecodeGraphPeerBridge requires two distinct devices; a capture cannot fork a stream "
            "into itself");
    }
    const ScopedDevice scope;
    cudaEvent_t created[3] = {nullptr, nullptr, nullptr};
    // fork_ is recorded on the origin, join_ and gate_ on the peer.
    const int device_of[3] = {origin_device, peer_device, peer_device};
    for (int slot = 0; slot < 3; ++slot) {
        cudaError_t status = cudaSetDevice(device_of[slot]);
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { (void)cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("DecodeGraphPeerBridge: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    fork_ = created[0];
    join_ = created[1];
    gate_ = created[2];
}

DecodeGraphPeerBridge::~DecodeGraphPeerBridge() {
    destroy_event(fork_);
    destroy_event(join_);
    destroy_event(gate_);
}

DecodeGraphPeerBridge::DecodeGraphPeerBridge(DecodeGraphPeerBridge&& other) noexcept
    : origin_device_(other.origin_device_), peer_device_(other.peer_device_), fork_(other.fork_),
      join_(other.join_), gate_(other.gate_) {
    other.fork_ = nullptr;
    other.join_ = nullptr;
    other.gate_ = nullptr;
}

DecodeGraphPeerBridge& DecodeGraphPeerBridge::operator=(DecodeGraphPeerBridge&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(fork_);
    destroy_event(join_);
    destroy_event(gate_);
    origin_device_ = other.origin_device_;
    peer_device_   = other.peer_device_;
    fork_          = other.fork_;
    join_          = other.join_;
    gate_          = other.gate_;
    other.fork_    = nullptr;
    other.join_    = nullptr;
    other.gate_    = nullptr;
    return *this;
}

void DecodeGraphPeerBridge::gate_launch(cudaStream_t peer_stream,
                                        cudaStream_t origin_stream) const {
    if (gate_ == nullptr) {
        throw std::logic_error("a moved-from DecodeGraphPeerBridge cannot gate a graph launch");
    }
    const ScopedDevice scope;
    ScopedDevice::set(peer_device_);
    CUDA_CHECK(cudaEventRecord(gate_, peer_stream));
    ScopedDevice::set(origin_device_);
    CUDA_CHECK(cudaStreamWaitEvent(origin_stream, gate_, 0));
}

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    capture(stream, body, DecodeGraphPeerCapture{});
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body,
                                    const DecodeGraphPeerCapture& peer) {
    const bool dual = peer.bridge != nullptr;
    if (dual) {
        if (!peer.bridge->live() || peer.stream == nullptr) {
            throw std::invalid_argument("dual-device capture requires a live peer bridge and "
                                        "the peer device's stream");
        }
    }
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    bool peer_forked = false;
    try {
        if (dual) {
            fork_peer(stream, peer);
            peer_forked = true;
        }
        body();
        if (dual) { join_peer(stream, peer); }
    } catch (...) {
        discard_capture(stream, dual ? &peer : nullptr, peer_forked);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

std::size_t DecodeGraphDefinition::node_count() const {
    if (graph_ == nullptr) { return 0; }
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes));
    return nodes;
}

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        CUDA_CHECK(err);
    }
    exec_ = exec;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer
