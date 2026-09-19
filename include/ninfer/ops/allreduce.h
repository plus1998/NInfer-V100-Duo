#pragma once

// ninfer::ops - two-device collectives.
//
// These are the cross-device primitives every tensor-parallel (tp == 2) forward pass needs: one
// summing reduction for row-parallel projections, one row gather for the vocabulary-split output
// head. Both operate on an ExecutionContext holding exactly two DeviceContexts and use each
// device's own compute stream; no stream parameter is accepted, because "the stream a device
// executes on" is already a property of its DeviceContext and a second, disagreeing stream story
// would break the ordering guarantees documented below.
//
// TRANSPORT. Payloads move with cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over unified virtual
// addresses, which every 64-bit Linux CUDA context has: a device pointer already names its
// device, so this one entry point expresses a cross-device transfer as well as a local one. When
// the driver grants peer access and startup validates it, the copy is a direct device-to-device
// PCIe transfer. Linux translated IOMMU domains (DMA/DMA-FQ) prohibit direct P2P regardless of
// advertised support or small-copy results. If either device uses such a domain, peer access is
// unavailable, or copies fail the exact startup check, both directions are disabled and CUDA
// stages the same copy through host memory. Startup validates
// that route too before allowing inference. Both qualified paths are stream-ordered, so callers
// need no peer-access branch; enable_peer_access() below reports which one is active.
//
// The equivalent cudaMemcpyPeerAsync entry point is deliberately NOT used: it is rejected inside a
// stream capture region (cudaErrorStreamCaptureUnsupported), which would make the whole
// tensor-parallel decode program uncapturable. See src/ops/common/allreduce.cu's pull_peer().
//
// ORDERING. Every transfer is a PULL: rank r reads the peer's source into storage that rank r
// alone owns, on rank r's own stream. Nothing a rank owns is ever written by the peer's stream, so
// the classic push hazard -- the peer overwriting a staging buffer that this rank has not finished
// reading yet -- cannot exist. Each rank issues two event records and two event waits:
//
//   stream(r):  ...producer of the rank-r inputs...
//               record(inputs_ready[r])
//               wait(inputs_ready[1-r])              // peer's source is complete
//               memcpyPeer(peer source -> rank-r storage)
//               record(pull_done[r])
//               wait(pull_done[1-r])                 // peer has finished reading MY source
//               ...local combine, for allreduce_sum...
//
// The two waits carry the whole correctness argument:
//
//   * wait(inputs_ready[1-r]) makes the inbound copy read a completed peer source, both within a
//     call and across calls -- in call k+1 that event is recorded after call k's local work on the
//     peer's stream, so the next pull cannot start before the previous call finished with the
//     source it reads.
//   * wait(pull_done[1-r]) is the write-after-read barrier for THIS rank's own source: it
//     guarantees the peer has finished reading rank r's buffer before rank r (or the caller, on
//     return) overwrites it. For allreduce_sum that protects the in-place combine; for both ops it
//     is what makes back-to-back calls that reuse the same buffers safe.
//
// POST-CONDITION. On return each rank's stream is ordered after the peer's read of that rank's
// inputs. Therefore an unbounded sequence of calls sharing the same buffers, staging, and
// PeerEvents needs NO host synchronization between calls -- which is exactly what a captured
// CUDA graph replays, and what a decode token's collectives do. For the 27B model that is
// 128 all-reduces (64 layers x 2 row-parallel projections: the mixer output and the MLP down
// projection) plus one allgather_rows per logit column, so 129 collectives at batch 1.
//
// Everything in a call is stream-ordered device work (memcpy, event record, event wait, kernel
// launch): no host round trip, no host-memory spin flag, no device or stream synchronization. That
// is what makes the sequence capturable. Measured: with both device streams enrolled in ONE
// capture -- the peer's stream joined to the origin's by an event fork -- the record/wait pairs
// above become graph EDGES rather than nodes, and the whole 128-collective decode program
// instantiates as a single cross-device cudaGraphExec. Event objects and staging storage are
// caller-owned and created once, because cudaEventCreate and cudaMalloc are not capturable.
//
// CALLER OBLIGATION -- the legacy default stream. These ops read their inputs on
// DeviceContext::stream, which core creates with cudaStreamNonBlocking and which therefore does
// NOT implicitly synchronize with a device's legacy default stream. Work issued with the plain
// cudaMemcpy/cudaMemset/<<<...>>> forms (no stream argument) lands on that default stream and is
// unordered against these collectives. A caller that stages inputs that way must retire them
// first (a device or stream synchronize, or an event); producing the inputs on
// DeviceContext::stream needs no extra step. This trap is invisible in single-device code, where
// the default stream orders everything.
//
// After a call returns the work is enqueued but not complete; the caller synchronizes each
// device's own stream, or lets subsequent stream-ordered work depend on it, as usual.

#include "core/device.h" // DeviceContext, ExecutionContext
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaEvent_t

#include <array>
#include <cstddef>

namespace ninfer::ops {

// Qualifies the actual cross-device copy route at startup. On Linux, first resolve both CUDA
// devices' PCI bus IDs to /sys/bus/pci/devices/<BDF>/iommu_group/type. DMA and DMA-FQ select the
// host-staged route without ever enabling direct P2P, even if a small probe could pass. Otherwise,
// when both directions advertise peer access, enable them and check two distinct 16 KiB patterns
// with the collectives' UVA D2D API on
// their destination compute streams. Returns true only when both copies are exact. A data
// mismatch or unavailable peer access disables both directions (including previously enabled
// access), verifies the same copies through CUDA's host-staged route, emits one diagnostic, and
// returns false. A failed staged check or CUDA API error throws; inference must not continue.
//
// Call once during setup, before graph capture or concurrent inference. This function allocates
// temporary probe buffers, synchronizes each compute stream, and releases its buffers while
// restoring the caller's current device. Peer-access changes are context-level operations;
// none of this setup is graph-capturable or belongs in a hot path. A non-TP2 context returns false.
bool enable_peer_access(const ExecutionContext& ec);

// The reusable cross-device ordering events: two per device, created on that device with timing
// disabled.
//
//   inputs_ready(r) - recorded on rank r's stream once the inputs rank r contributes are complete;
//                     the peer waits on it before reading them.
//   pull_done(r)    - recorded on rank r's stream once rank r's inbound copy has finished reading
//                     the peer's source; the peer waits on it before overwriting that source.
//
// Re-recording an event a previous call already recorded is well defined: cudaStreamWaitEvent
// captures the event's state at the time the wait is issued, and the collectives always issue a
// call's waits before the next call's records, so one instance serves an unbounded number of
// sequential calls. Instances are not thread-safe: one instance belongs to one stream pair. A
// moved-from instance holds no events and must not be passed to a collective (the ops reject it).
class PeerEvents {
public:
    explicit PeerEvents(const ExecutionContext& ec);
    ~PeerEvents();

    PeerEvents(const PeerEvents&)            = delete;
    PeerEvents& operator=(const PeerEvents&) = delete;
    PeerEvents(PeerEvents&& other) noexcept;
    PeerEvents& operator=(PeerEvents&& other) noexcept;

    [[nodiscard]] cudaEvent_t inputs_ready(int rank) const noexcept {
        return inputs_ready_[static_cast<std::size_t>(rank)];
    }

    [[nodiscard]] cudaEvent_t pull_done(int rank) const noexcept {
        return pull_done_[static_cast<std::size_t>(rank)];
    }

    // False for a moved-from instance.
    [[nodiscard]] bool live() const noexcept {
        return inputs_ready_[0] != nullptr && inputs_ready_[1] != nullptr &&
               pull_done_[0] != nullptr && pull_done_[1] != nullptr;
    }

private:
    std::array<cudaEvent_t, 2> inputs_ready_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> pull_done_{nullptr, nullptr};
};

/**
 * Two-device summing all-reduce, in place:
 *
 *   ideal[i] = buffer_rank0[i] + buffer_rank1[i]   for every i,
 *
 * written back to both `buffer[0]` and `buffer[1]`, which afterwards hold the identical sum.
 *
 * `buffer[r]` is a contiguous BF16 tensor resident on `ec.dev[r]`; both ranks carry the same
 * shape. `staging[r]` is scratch of the same dtype and shape, also resident on `ec.dev[r]`, and
 * must not overlap `buffer[r]`; it receives the peer's contribution and its contents after the
 * call are unspecified. Rank r's stream is the only stream that ever touches `staging[r]`.
 *
 * The oracle evaluates `ideal` in FP64 from the represented BF16 inputs; the observable BF16
 * output is that value rounded once to BF16 storage, and the local combine is the qualified
 * residual_add computation body (FP32 accumulation of the two BF16 operands, one
 * round-to-nearest-even on store). The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp == 2` and a live `events`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events);

/**
 * Two-device row gather, exact (no arithmetic). The gathered axis is `ne[1]`, so each rank
 * contributes one contiguous block of a `[C, R]` tensor (`ne[0] == C` is the row length,
 * `ne[1] == R` the row count, `ne[2] == ne[3] == 1`):
 *
 *   destination[r][c, row] = part[0][c, row]                    for row <  part[0].ne[1]
 *   destination[r][c, row] = part[1][c, row - part[0].ne[1]]    otherwise
 *
 * for both r, so each device ends up holding the identical full image. `part[r]` is the row range
 * owned by `ec.dev[r]`: rank 0 owns the leading rows and rank 1 the trailing rows, and the two
 * counts must sum to the destination row count. `destination[r]` and `part[r]` are contiguous,
 * share one dtype, agree on `ne[0]`, live on `ec.dev[r]`, and must not overlap. A caller whose
 * split axis is not `ne[1]` (for example `[V, B]` logits split by vocabulary with B > 1) arranges
 * the layout so that it is; this Op relocates contiguous storage and performs no transpose.
 *
 * `destination[r]` is written only by rank r's stream: rank r copies its own block locally and
 * pulls the peer's block. The transformation is a pure relocation of storage, so it is verified by
 * exact byte comparison. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp == 2` and a live `events`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
