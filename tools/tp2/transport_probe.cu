// The transport ladder behind the one-line transport change in
// src/ops/common/allreduce.cu.
//
// tools/tp2/capture_probe.cu established that a TP2 decode program CAN be
// captured across two devices, but that the transport the collectives used at
// the time -- cudaMemcpyPeerAsync, issued on the DESTINATION device's stream --
// is rejected inside a capture region with cudaErrorStreamCaptureUnsupported.
// This tool enumerates the alternatives and reports, for each, the two things
// that matter:
//
//   EAGER  does it actually move the right bytes across devices, with peer
//          access DISABLED (the GeForce case, where CUDA stages via host)?
//   GRAPH  is it accepted inside a cross-device capture region?
//
// Candidates:
//   1. cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)  over UVA pointers
//   2. cudaMemcpyAsync(..., cudaMemcpyDefault)         over UVA pointers
//   3. cudaGraphAddMemcpyNode spliced directly into the capturing graph via
//      cudaStreamGetCaptureInfo + cudaStreamUpdateCaptureDependencies
//   (cudaMemcpyPeerAsync is measured too, as the control that motivated this.)
//
// WHAT IT ESTABLISHED (2x RTX 5090, CUDA 13.1, driver 580.178.04, no P2P):
// all three candidates pass both columns; cudaMemcpyPeerAsync passes EAGER and
// fails GRAPH. Candidate 1 was chosen -- it is a one-token change at one call
// site, keeps the copy on the destination stream (so the event-ordering
// post-condition documented in include/ninfer/ops/allreduce.h's contract header
// -- on return each rank's stream is ordered after the peer's read of that
// rank's inputs -- is untouched), and needs no capture-aware branch.
// Under unified virtual addressing a device pointer already names its device,
// so it expresses the same cross-device transfer as the peer form.
//
// Re-run on any new two-GPU topology, in particular where peer access IS
// granted: this machine cannot exercise the direct-P2P path.
//
// This is a standalone tool, not wired into the CMake build, following the
// same standalone-nvcc convention as tools/tp2/p2p_probe.cu.
//
// Build:
//   nvcc -arch=sm_120a -O2 tools/tp2/transport_probe.cu -o /tmp/transport_probe
//
// Run:
//   /tmp/transport_probe [dev_a] [dev_b]   # defaults: 0 1
//
// Exit codes:
//   0 = at least one candidate is both eager-correct and capturable, so the
//       collectives have a capture-legal transport on this platform.
//   1 = a hard CUDA failure.
//   2 = no candidate is both eager-correct and capturable. The collectives
//       cannot be captured here and the TP2 decode path must stay eager.
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

namespace {

constexpr int kElems = 4096;
constexpr size_t kBytes = kElems * sizeof(float);

struct Pair {
  int device[2] = {0, 1};
  cudaStream_t stream[2] = {nullptr, nullptr};
  float* source = nullptr;       // on device[1]
  float* destination = nullptr;  // on device[0]
};

struct Candidate {
  const char* name;
  bool eager_ok = false;
  bool graph_ok = false;
};

// The reference pattern the destination must end up holding.
float expected(int index) { return 1.0f + static_cast<float>(index); }

void reset_destination(Pair& p) {
  CK(cudaSetDevice(p.device[0]));
  CK(cudaMemset(p.destination, 0, kBytes));
  CK(cudaDeviceSynchronize());
}

bool destination_is_correct(Pair& p) {
  std::vector<float> host(kElems);
  CK(cudaSetDevice(p.device[0]));
  CK(cudaMemcpy(host.data(), p.destination, kBytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kElems; ++i) {
    if (host[i] != expected(i)) {
      printf("      mismatch at %d: got %.4f want %.4f\n", i, host[i],
             expected(i));
      return false;
    }
  }
  return true;
}

// --- eager column -----------------------------------------------------------

bool eager_copy(Pair& p, cudaError_t (*issue)(Pair&)) {
  reset_destination(p);
  CK(cudaSetDevice(p.device[0]));
  const cudaError_t status = issue(p);
  printf("      eager issue -> %s\n", cudaGetErrorName(status));
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  CK(cudaStreamSynchronize(p.stream[0]));
  return destination_is_correct(p);
}

cudaError_t issue_peer(Pair& p) {
  return cudaMemcpyPeerAsync(p.destination, p.device[0], p.source, p.device[1],
                             kBytes, p.stream[0]);
}

cudaError_t issue_d2d(Pair& p) {
  return cudaMemcpyAsync(p.destination, p.source, kBytes,
                         cudaMemcpyDeviceToDevice, p.stream[0]);
}

cudaError_t issue_default(Pair& p) {
  return cudaMemcpyAsync(p.destination, p.source, kBytes, cudaMemcpyDefault,
                         p.stream[0]);
}

// --- graph column -----------------------------------------------------------

// Opens a cross-device capture, runs `issue` inside it, and reports whether the
// call was accepted. When it was, the capture is completed and replayed so the
// bytes are checked too -- "accepted" is not the same as "works".
bool capturable(Pair& p, cudaError_t (*issue)(Pair&), cudaEvent_t fork_event,
                cudaEvent_t join_event) {
  reset_destination(p);
  CK(cudaSetDevice(p.device[0]));
  CK(cudaStreamBeginCapture(p.stream[0], cudaStreamCaptureModeThreadLocal));
  CK(cudaEventRecord(fork_event, p.stream[0]));
  CK(cudaSetDevice(p.device[1]));
  CK(cudaStreamWaitEvent(p.stream[1], fork_event, 0));

  CK(cudaSetDevice(p.device[0]));
  const cudaError_t status = issue(p);
  printf("      capture issue -> %s\n", cudaGetErrorName(status));

  CK(cudaSetDevice(p.device[1]));
  cudaEventRecord(join_event, p.stream[1]);
  CK(cudaSetDevice(p.device[0]));
  cudaStreamWaitEvent(p.stream[0], join_event, 0);

  cudaGraph_t graph = nullptr;
  const cudaError_t end = cudaStreamEndCapture(p.stream[0], &graph);
  if (status != cudaSuccess || end != cudaSuccess || graph == nullptr) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiated = cudaGraphInstantiate(&exec, graph, 0);
  if (instantiated != cudaSuccess) {
    printf("      instantiate -> %s\n", cudaGetErrorName(instantiated));
    cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  CK(cudaGraphLaunch(exec, p.stream[0]));
  CK(cudaStreamSynchronize(p.stream[0]));
  const bool correct = destination_is_correct(p);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  return correct;
}

// Candidate 3 does not go through a stream call at all: it builds the memcpy
// node itself and splices it into the graph the capture is accumulating.
bool capturable_graph_node(Pair& p, cudaEvent_t fork_event,
                           cudaEvent_t join_event) {
  reset_destination(p);
  CK(cudaSetDevice(p.device[0]));
  CK(cudaStreamBeginCapture(p.stream[0], cudaStreamCaptureModeThreadLocal));
  CK(cudaEventRecord(fork_event, p.stream[0]));
  CK(cudaSetDevice(p.device[1]));
  CK(cudaStreamWaitEvent(p.stream[1], fork_event, 0));
  CK(cudaSetDevice(p.device[0]));

  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaGraph_t capturing = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  size_t dependency_count = 0;
  unsigned long long id = 0;
  const cudaError_t info =
      cudaStreamGetCaptureInfo(p.stream[0], &status, &id, &capturing,
                               &dependencies, nullptr, &dependency_count);
  printf("      cudaStreamGetCaptureInfo -> %s (%zu deps)\n",
         cudaGetErrorName(info), dependency_count);

  cudaMemcpy3DParms params = {};
  params.srcPtr = make_cudaPitchedPtr(p.source, kBytes, kBytes, 1);
  params.dstPtr = make_cudaPitchedPtr(p.destination, kBytes, kBytes, 1);
  params.extent = make_cudaExtent(kBytes, 1, 1);
  params.kind = cudaMemcpyDeviceToDevice;

  cudaGraphNode_t node = nullptr;
  cudaError_t added = cudaGraphAddMemcpyNode(&node, capturing, dependencies,
                                             dependency_count, &params);
  printf("      cudaGraphAddMemcpyNode(D2D) -> %s\n", cudaGetErrorName(added));
  if (added != cudaSuccess) {
    cudaGetLastError();
    params.kind = cudaMemcpyDefault;
    added = cudaGraphAddMemcpyNode(&node, capturing, dependencies,
                                   dependency_count, &params);
    printf("      cudaGraphAddMemcpyNode(Default) -> %s\n",
           cudaGetErrorName(added));
  }
  if (added == cudaSuccess) {
    const cudaError_t updated = cudaStreamUpdateCaptureDependencies(
        p.stream[0], &node, nullptr, 1, cudaStreamSetCaptureDependencies);
    printf("      cudaStreamUpdateCaptureDependencies -> %s\n",
           cudaGetErrorName(updated));
    if (updated != cudaSuccess) added = updated;
  }

  CK(cudaSetDevice(p.device[1]));
  cudaEventRecord(join_event, p.stream[1]);
  CK(cudaSetDevice(p.device[0]));
  cudaStreamWaitEvent(p.stream[0], join_event, 0);

  cudaGraph_t graph = nullptr;
  const cudaError_t end = cudaStreamEndCapture(p.stream[0], &graph);
  if (added != cudaSuccess || end != cudaSuccess || graph == nullptr) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiated = cudaGraphInstantiate(&exec, graph, 0);
  if (instantiated != cudaSuccess) {
    printf("      instantiate -> %s\n", cudaGetErrorName(instantiated));
    cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  CK(cudaGraphLaunch(exec, p.stream[0]));
  CK(cudaStreamSynchronize(p.stream[0]));
  const bool correct = destination_is_correct(p);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  return correct;
}

int transport_probe(int dev_a, int dev_b) {
  Pair p;
  p.device[0] = dev_a;
  p.device[1] = dev_b;
  for (int i = 0; i < 2; ++i) {
    CK(cudaSetDevice(p.device[i]));
    CK(cudaStreamCreateWithFlags(&p.stream[i], cudaStreamNonBlocking));
  }
  CK(cudaSetDevice(p.device[0]));
  CK(cudaMalloc(&p.destination, kBytes));
  CK(cudaSetDevice(p.device[1]));
  CK(cudaMalloc(&p.source, kBytes));
  std::vector<float> host(kElems);
  for (int i = 0; i < kElems; ++i) host[i] = expected(i);
  CK(cudaMemcpy(p.source, host.data(), kBytes, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());

  int can_ab = 0;
  int can_ba = 0;
  CK(cudaDeviceCanAccessPeer(&can_ab, dev_a, dev_b));
  CK(cudaDeviceCanAccessPeer(&can_ba, dev_b, dev_a));
  printf("devices %d,%d  peer access %d/%d (0 = CUDA stages copies via host)\n",
         dev_a, dev_b, can_ab, can_ba);
  printf("moving %zu B from dev%d -> dev%d, issued on dev%d's stream\n\n",
         kBytes, dev_b, dev_a, dev_a);

  cudaEvent_t fork_event = nullptr;
  cudaEvent_t join_event = nullptr;
  CK(cudaSetDevice(p.device[0]));
  CK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
  CK(cudaSetDevice(p.device[1]));
  CK(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));

  std::vector<Candidate> results;

  printf("[control] cudaMemcpyPeerAsync\n");
  Candidate peer{"cudaMemcpyPeerAsync"};
  peer.eager_ok = eager_copy(p, issue_peer);
  peer.graph_ok = capturable(p, issue_peer, fork_event, join_event);
  results.push_back(peer);

  printf("\n[1] cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over UVA\n");
  Candidate d2d{"cudaMemcpyAsync(D2D)"};
  d2d.eager_ok = eager_copy(p, issue_d2d);
  d2d.graph_ok = capturable(p, issue_d2d, fork_event, join_event);
  results.push_back(d2d);

  printf("\n[2] cudaMemcpyAsync(cudaMemcpyDefault) over UVA\n");
  Candidate any{"cudaMemcpyAsync(Default)"};
  any.eager_ok = eager_copy(p, issue_default);
  any.graph_ok = capturable(p, issue_default, fork_event, join_event);
  results.push_back(any);

  printf("\n[3] cudaGraphAddMemcpyNode spliced into the capturing graph\n");
  Candidate node{"cudaGraphAddMemcpyNode"};
  // No eager form exists -- a graph node is only meaningful inside a graph --
  // so the eager column is inherited from the D2D parameters it is built from.
  node.eager_ok = d2d.eager_ok;
  node.graph_ok = capturable_graph_node(p, fork_event, join_event);
  results.push_back(node);

  printf("\n%-28s %-8s %-8s\n", "transport", "EAGER", "GRAPH");
  bool any_viable = false;
  for (const Candidate& c : results) {
    printf("%-28s %-8s %-8s\n", c.name, c.eager_ok ? "ok" : "FAIL",
           c.graph_ok ? "ok" : "FAIL");
    if (c.eager_ok && c.graph_ok) any_viable = true;
  }
  if (!any_viable) {
    printf("\nNO capture-legal cross-device transport on this platform: the "
           "collectives cannot be captured here\n");
    return 2;
  }
  printf("\nPASS: at least one transport is both eager-correct and "
         "capturable\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
    printf("skip: transport probe needs two CUDA devices\n");
    return 77;
  }
  const int a = argc > 1 ? atoi(argv[1]) : 0;
  const int b = argc > 2 ? atoi(argv[2]) : 1;
  return transport_probe(a, b);
}
