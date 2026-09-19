// Answers the question that had to be settled before a dual-device CUDA Graph
// capture path could be written at all: can the TP2 decode program -- work on
// TWO devices' streams, ordered against each other by the pull-based
// all-reduce's cross-device events -- be captured into a CUDA graph, and does
// the resulting graph replay to the same values as eager execution?
//
// It reproduces the real choreography in miniature: two devices, one
// non-blocking stream each, the four-event inputs_ready/pull_done protocol
// from include/ninfer/ops/allreduce.h verbatim in structure, three rounds of
// it inside one capture region, plus a local kernel per rank per round.
//
// WHAT IT ESTABLISHED (2x RTX 5090, CUDA 13.1, driver 580.178.04, no P2P):
//   * A capture begun on device 0's stream CAN enroll device 1's stream, by
//     recording an event on the origin and waiting it on the peer. Both
//     streams then report the SAME capture id.
//   * cudaMemcpyPeerAsync is REJECTED inside a capture region with
//     cudaErrorStreamCaptureUnsupported. This is why src/ops/common/
//     allreduce.cu pulls with cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over
//     UVA pointers instead; see tools/tp2/transport_probe.cu for the ladder of
//     alternatives that was tried.
//   * Two INDEPENDENT captures cannot be linked: an event wait crossing them
//     fails with cudaErrorStreamCaptureMerge. A tensor-parallel decode program
//     is therefore ONE graph holding both devices' nodes, not one graph per
//     device.
//   * The cross-device event records and waits become graph EDGES, not nodes.
//     The captured graph holds only real device work, on both devices.
//
// Re-run this on any new two-GPU topology before trusting the capture path
// there -- in particular on hardware where peer access IS granted, which this
// machine cannot exercise.
//
// This is a standalone tool, not wired into the CMake build, following the
// same standalone-nvcc convention as tools/tp2/p2p_probe.cu.
//
// Build:
//   nvcc -arch=sm_120a -O2 tools/tp2/capture_probe.cu -o /tmp/capture_probe
//
// Run:
//   /tmp/capture_probe [dev_a] [dev_b]   # defaults: 0 1
//
// Exit codes:
//   0 = cross-device capture works: the peer stream enrolled, a capture-legal
//       transport was found, the graph instantiated, and replay reproduced the
//       eager values twice. The dual-device capture path is viable here.
//   1 = a hard CUDA failure, or the graph replayed DIFFERENT values from eager
//       (a correctness failure, not a capability one).
//   2 = cross-device capture is not viable on this platform: either the peer
//       stream could not be enrolled, or no capture-legal transport exists.
//       Everything measured up to that point is printed above the exit. This
//       is the finding that would force the TP2 decode path to stay eager.
//  77 = fewer than two CUDA devices visible (nothing to probe).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA %s:%d: %s: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorName(e), cudaGetErrorString(e));                   \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// Reports a call's status instead of aborting on it: the whole point of this
// tool is to learn WHICH calls a capture region rejects.
#define SOFT(label, x)                                                       \
  do {                                                                       \
    last_status = (x);                                                       \
    printf("  %-56s -> %s\n", label, cudaGetErrorName(last_status));         \
  } while (0)

namespace {

constexpr int kElems = 4096;
constexpr size_t kBytes = kElems * sizeof(float);
constexpr int kBlock = 256;
constexpr int kGrid = (kElems + kBlock - 1) / kBlock;
constexpr int kRounds = 3;

cudaError_t last_status = cudaSuccess;

struct Rank {
  int device = 0;
  cudaStream_t stream = nullptr;
  float* buffer = nullptr;
  float* staging = nullptr;
};

__global__ void add_into(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void scale_by(float* dst, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] *= k;
}

// The inbound half of a pull, in the form src/ops/common/allreduce.cu uses:
// issued on the DESTINATION's stream, with the source resident on the peer.
// cudaMemcpyPeerAsync is the form this probe proved uncapturable.
cudaError_t pull(void* dst, const void* src, cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, kBytes, cudaMemcpyDeviceToDevice, stream);
}

void setup(Rank r[2], int dev_a, int dev_b) {
  r[0].device = dev_a;
  r[1].device = dev_b;
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    CK(cudaStreamCreateWithFlags(&r[i].stream, cudaStreamNonBlocking));
    CK(cudaMalloc(&r[i].buffer, kBytes));
    CK(cudaMalloc(&r[i].staging, kBytes));
  }
}

// Both ranks start from a distinct, position-dependent pattern so a dropped
// rank or a swapped buffer cannot coincidentally produce the right answer.
void fill(Rank r[2]) {
  std::vector<float> host(kElems);
  for (int i = 0; i < 2; ++i) {
    const float base = i == 0 ? 10.0f : 100.0f;
    for (int j = 0; j < kElems; ++j) host[j] = base + static_cast<float>(j % 7);
    CK(cudaSetDevice(r[i].device));
    CK(cudaMemcpy(r[i].buffer, host.data(), kBytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
  }
}

std::vector<float> read_back(const Rank& rank) {
  std::vector<float> host(kElems);
  CK(cudaSetDevice(rank.device));
  CK(cudaMemcpy(host.data(), rank.buffer, kBytes, cudaMemcpyDeviceToHost));
  return host;
}

void sync_both(Rank r[2]) {
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    CK(cudaStreamSynchronize(r[i].stream));
  }
}

// One round of the real schedule: per-rank local work, then the three-phase
// all-reduce. `soft` selects whether failures abort (eager) or are reported
// (inside a capture region, where a rejection is the result being measured).
bool issue_round(Rank r[2], cudaEvent_t inputs_ready[2], cudaEvent_t pull_done[2],
                 bool soft) {
  for (int rank = 0; rank < 2; ++rank) {
    CK(cudaSetDevice(r[rank].device));
    scale_by<<<kGrid, kBlock, 0, r[rank].stream>>>(r[rank].buffer, 0.5f, kElems);
    if (soft) {
      SOFT("kernel launch", cudaGetLastError());
      if (last_status != cudaSuccess) return false;
    } else {
      CK(cudaGetLastError());
    }
  }
  // Phase A: publish "my operand is complete", before any wait observes it.
  for (int rank = 0; rank < 2; ++rank) {
    CK(cudaSetDevice(r[rank].device));
    if (soft) {
      SOFT("cudaEventRecord(inputs_ready)",
           cudaEventRecord(inputs_ready[rank], r[rank].stream));
      if (last_status != cudaSuccess) return false;
    } else {
      CK(cudaEventRecord(inputs_ready[rank], r[rank].stream));
    }
  }
  // Phase B: each rank pulls the peer's operand into storage only it owns.
  for (int rank = 0; rank < 2; ++rank) {
    CK(cudaSetDevice(r[rank].device));
    if (soft) {
      SOFT("cudaStreamWaitEvent(peer inputs_ready)",
           cudaStreamWaitEvent(r[rank].stream, inputs_ready[1 - rank], 0));
      if (last_status != cudaSuccess) return false;
      SOFT("pull peer -> own staging (on DESTINATION stream)",
           pull(r[rank].staging, r[1 - rank].buffer, r[rank].stream));
      if (last_status != cudaSuccess) return false;
      SOFT("cudaEventRecord(pull_done)",
           cudaEventRecord(pull_done[rank], r[rank].stream));
      if (last_status != cudaSuccess) return false;
    } else {
      CK(cudaStreamWaitEvent(r[rank].stream, inputs_ready[1 - rank], 0));
      CK(pull(r[rank].staging, r[1 - rank].buffer, r[rank].stream));
      CK(cudaEventRecord(pull_done[rank], r[rank].stream));
    }
  }
  // Phase C: the in-place combine may only overwrite the buffer once the peer
  // has finished reading it.
  for (int rank = 0; rank < 2; ++rank) {
    CK(cudaSetDevice(r[rank].device));
    if (soft) {
      SOFT("cudaStreamWaitEvent(peer pull_done)",
           cudaStreamWaitEvent(r[rank].stream, pull_done[1 - rank], 0));
      if (last_status != cudaSuccess) return false;
    } else {
      CK(cudaStreamWaitEvent(r[rank].stream, pull_done[1 - rank], 0));
    }
    add_into<<<kGrid, kBlock, 0, r[rank].stream>>>(r[rank].buffer,
                                                   r[rank].staging, kElems);
    if (soft) {
      SOFT("kernel launch", cudaGetLastError());
      if (last_status != cudaSuccess) return false;
    } else {
      CK(cudaGetLastError());
    }
  }
  return true;
}

void print_node_histogram(cudaGraph_t graph) {
  size_t count = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  if (count != 0) CK(cudaGraphGetNodes(graph, nodes.data(), &count));
  int kernel = 0, memcpy_n = 0, other = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    CK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeKernel) {
      ++kernel;
    } else if (type == cudaGraphNodeTypeMemcpy) {
      ++memcpy_n;
    } else {
      ++other;
    }
  }
  // Expected for kRounds rounds x 2 ranks: 2 kernels and 1 memcpy per rank per
  // round, and ZERO event nodes -- the record/wait pairs are edges.
  printf("  captured graph: %zu nodes (kernel=%d memcpy=%d other=%d); "
         "expected kernel=%d memcpy=%d other=0\n",
         count, kernel, memcpy_n, other, kRounds * 2 * 2, kRounds * 2);
}

// The negative control. It is what proves the single-graph architecture is
// forced rather than merely chosen: if two independent captures COULD be
// linked, one graph per device would be viable.
void report_independent_capture_control(Rank r[2], cudaEvent_t probe_event) {
  printf("\n[control] two independent captures, one event wait crossing them\n");
  cudaGraph_t g0 = nullptr;
  cudaGraph_t g1 = nullptr;
  CK(cudaSetDevice(r[0].device));
  SOFT("cudaStreamBeginCapture(dev0)",
       cudaStreamBeginCapture(r[0].stream, cudaStreamCaptureModeThreadLocal));
  CK(cudaSetDevice(r[1].device));
  SOFT("cudaStreamBeginCapture(dev1)",
       cudaStreamBeginCapture(r[1].stream, cudaStreamCaptureModeThreadLocal));
  CK(cudaSetDevice(r[0].device));
  SOFT("cudaEventRecord(dev0 stream)",
       cudaEventRecord(probe_event, r[0].stream));
  CK(cudaSetDevice(r[1].device));
  SOFT("cudaStreamWaitEvent(dev1 stream, dev0 event)  <-- the answer",
       cudaStreamWaitEvent(r[1].stream, probe_event, 0));
  const bool merged = last_status == cudaSuccess;
  CK(cudaSetDevice(r[0].device));
  cudaStreamEndCapture(r[0].stream, &g0);
  CK(cudaSetDevice(r[1].device));
  cudaStreamEndCapture(r[1].stream, &g1);
  if (g0 != nullptr) cudaGraphDestroy(g0);
  if (g1 != nullptr) cudaGraphDestroy(g1);
  cudaGetLastError();
  printf("  => independent per-device captures are %s\n",
         merged ? "LINKABLE (one graph per device would be viable)"
                : "NOT linkable; a TP2 decode program must be ONE graph");
}

int capture_probe(int dev_a, int dev_b) {
  Rank r[2];
  setup(r, dev_a, dev_b);

  int can_ab = 0;
  int can_ba = 0;
  CK(cudaDeviceCanAccessPeer(&can_ab, dev_a, dev_b));
  CK(cudaDeviceCanAccessPeer(&can_ba, dev_b, dev_a));
  printf("devices %d,%d  peer access %d/%d (0 = CUDA stages copies via host)\n",
         dev_a, dev_b, can_ab, can_ba);

  cudaEvent_t inputs_ready[2];
  cudaEvent_t pull_done[2];
  cudaEvent_t fork_event = nullptr;
  cudaEvent_t join_event = nullptr;
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(r[i].device));
    CK(cudaEventCreateWithFlags(&inputs_ready[i], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&pull_done[i], cudaEventDisableTiming));
  }
  CK(cudaSetDevice(r[0].device));
  CK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
  CK(cudaSetDevice(r[1].device));
  CK(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));

  // ---------------------------------------------------------------- eager
  printf("\n[eager] the same choreography, executed directly\n");
  fill(r);
  for (int round = 0; round < kRounds; ++round) {
    issue_round(r, inputs_ready, pull_done, /*soft=*/false);
  }
  sync_both(r);
  const std::vector<float> eager_0 = read_back(r[0]);
  const std::vector<float> eager_1 = read_back(r[1]);
  printf("  ok: dev%d[0]=%.4f dev%d[0]=%.4f\n", dev_a, eager_0[0], dev_b,
         eager_1[0]);

  // ---------------------------------------------------------------- capture
  printf("\n[capture] origin = dev%d stream; dev%d stream forked in\n", dev_a,
         dev_b);
  CK(cudaSetDevice(r[0].device));
  SOFT("cudaStreamBeginCapture(origin)",
       cudaStreamBeginCapture(r[0].stream, cudaStreamCaptureModeThreadLocal));
  if (last_status != cudaSuccess) return 2;

  SOFT("cudaEventRecord(fork, origin stream)",
       cudaEventRecord(fork_event, r[0].stream));
  if (last_status != cudaSuccess) return 2;
  CK(cudaSetDevice(r[1].device));
  SOFT("cudaStreamWaitEvent(peer stream, fork)  <-- enrolls the peer",
       cudaStreamWaitEvent(r[1].stream, fork_event, 0));
  if (last_status != cudaSuccess) {
    printf("  => the peer stream cannot be enrolled; cross-device capture is "
           "unavailable here\n");
    cudaGraph_t discard = nullptr;
    CK(cudaSetDevice(r[0].device));
    cudaStreamEndCapture(r[0].stream, &discard);
    if (discard != nullptr) cudaGraphDestroy(discard);
    return 2;
  }

  unsigned long long id_origin = 0;
  unsigned long long id_peer = 0;
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  CK(cudaStreamGetCaptureInfo(r[1].stream, &status, &id_peer));
  const int peer_active = static_cast<int>(status);
  CK(cudaSetDevice(r[0].device));
  CK(cudaStreamGetCaptureInfo(r[0].stream, &status, &id_origin));
  printf("  peer capture status=%d, capture ids %llu / %llu, same graph=%d\n",
         peer_active, id_origin, id_peer,
         static_cast<int>(id_origin == id_peer));

  bool body_ok = true;
  for (int round = 0; round < kRounds && body_ok; ++round) {
    body_ok = issue_round(r, inputs_ready, pull_done, /*soft=*/true);
  }
  if (!body_ok) {
    printf("  => the decode body is not capturable with this transport; see "
           "tools/tp2/transport_probe.cu\n");
    cudaGraph_t discard = nullptr;
    CK(cudaSetDevice(r[0].device));
    cudaStreamEndCapture(r[0].stream, &discard);
    if (discard != nullptr) cudaGraphDestroy(discard);
    return 2;
  }

  // The join is mandatory: without it cudaStreamEndCapture reports
  // cudaErrorStreamCaptureUnjoined.
  CK(cudaSetDevice(r[1].device));
  SOFT("cudaEventRecord(join, peer stream)",
       cudaEventRecord(join_event, r[1].stream));
  if (last_status != cudaSuccess) return 2;
  CK(cudaSetDevice(r[0].device));
  SOFT("cudaStreamWaitEvent(origin stream, join)  <-- rejoins the peer",
       cudaStreamWaitEvent(r[0].stream, join_event, 0));
  if (last_status != cudaSuccess) return 2;

  cudaGraph_t graph = nullptr;
  SOFT("cudaStreamEndCapture", cudaStreamEndCapture(r[0].stream, &graph));
  if (last_status != cudaSuccess || graph == nullptr) return 2;
  print_node_histogram(graph);

  cudaGraphExec_t exec = nullptr;
  SOFT("cudaGraphInstantiate", cudaGraphInstantiate(&exec, graph, 0));
  if (last_status != cudaSuccess) return 2;

  // ---------------------------------------------------------------- replay
  printf("\n[replay] one launch on the origin stream drives BOTH devices\n");
  fill(r);
  CK(cudaSetDevice(r[0].device));
  SOFT("cudaGraphLaunch(origin stream)", cudaGraphLaunch(exec, r[0].stream));
  if (last_status != cudaSuccess) return 1;
  sync_both(r);
  const std::vector<float> replay_0 = read_back(r[0]);
  const std::vector<float> replay_1 = read_back(r[1]);

  fill(r);
  CK(cudaSetDevice(r[0].device));
  CK(cudaGraphLaunch(exec, r[0].stream));
  sync_both(r);
  const std::vector<float> again_0 = read_back(r[0]);
  const std::vector<float> again_1 = read_back(r[1]);

  const bool matches_eager = replay_0 == eager_0 && replay_1 == eager_1;
  const bool reproducible = again_0 == replay_0 && again_1 == replay_1;
  printf("  replay == eager: %s\n", matches_eager ? "YES" : "NO");
  printf("  replay 2 == replay 1: %s\n", reproducible ? "YES" : "NO");

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);

  report_independent_capture_control(r, fork_event);

  if (!matches_eager || !reproducible) {
    printf("\nFAIL: cross-device capture is supported but does not reproduce "
           "eager execution\n");
    return 1;
  }
  printf("\nPASS: cross-device capture works -- peer stream enrolled in one "
         "graph, replay reproduces eager, twice\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
    printf("skip: cross-device capture probe needs two CUDA devices\n");
    return 77;
  }
  const int a = argc > 1 ? atoi(argv[1]) : 0;
  const int b = argc > 2 ? atoi(argv[2]) : 1;
  return capture_probe(a, b);
}
