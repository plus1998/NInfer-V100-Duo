// Verifies PCIe P2P is available between two GPUs (the two RTX 5090s used
// for TP2) and measures one-way bandwidth in both directions. When P2P is
// unavailable, measures the automatic host-staged transport instead of
// aborting -- a cross-device copy works without peer access because CUDA
// silently stages it through host memory, and the TP2 all-reduce design
// issues that copy regardless of whether true P2P is available (P2P is an
// optimization when present, not a hard requirement).
//
// NOTE ON THE TRANSPORT MEASURED HERE. This probe times cudaMemcpyPeerAsync,
// which is what the TP2 collectives used when it was written. They now issue
// cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over UVA pointers instead, because
// stream capture rejects the peer entry point (see
// tools/tp2/transport_probe.cu). The two forms express the same transfer and
// measured indistinguishably, so these bandwidth and latency numbers still
// characterize the link -- but a like-for-like transport comparison belongs to
// transport_probe.cu, not here.
//
// This is a standalone tool, not wired into the CMake build -- tools/ has
// no CMakeLists.txt of its own and the top-level CMakeLists.txt only adds
// src/apps/tests/bench. This follows the same standalone-nvcc convention as
// tools/hbm_bandwidth_probe.cu.
//
// Build:
//   nvcc -arch=sm_120a -O2 tools/tp2/p2p_probe.cu -o /tmp/p2p_probe
//
// Run:
//   /tmp/p2p_probe [dev_a] [dev_b]   # defaults: 0 1
//
// Always measured, both directions: a 256 MiB copy (3 warm-up + 8 timed
// cudaMemcpyPeerAsync copies, event-timed) for one-way bandwidth in GiB/s,
// and a 10 KiB copy (10 warm-up + 1000 timed, stream-serialized,
// event-timed) for mean per-copy latency in microseconds -- the small-
// transfer case dominates the TP2 all-reduce (128 x ~10 KiB per decode
// token: 64 layers x 2 row-parallel projections).
//
// Exit codes:
//   0 = true P2P available (canAccess=1 both directions) AND big-copy
//       bandwidth >= 25 GiB/s both directions.
//   1 = P2P available but bandwidth threshold not met, or a hard CUDA
//       failure (the CK macro below also exits 1 on any unexpected error).
//   2 = P2P unavailable (canAccess=0 in either direction); the staged
//       (host-relayed) transport was measured instead and all numbers are
//       printed above this exit. This is not a hard failure -- the TP2
//       plan proceeds via cudaMemcpyPeerAsync's automatic staging, with
//       the measured overhead replacing the design's ≤2% estimate.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                \
  do {                                                                       \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(e));                                        \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr size_t kBigBytes = 256u << 20;  // 256 MiB
constexpr int kBigWarmup = 3;
constexpr int kBigTimed = 8;
constexpr double kMinGiBps = 25.0;

constexpr size_t kSmallBytes = 10u << 10;  // 10 KiB
constexpr int kSmallWarmup = 10;
constexpr int kSmallTimed = 1000;

// Enables peer access from device `from` onto device `to`, treating
// cudaErrorPeerAccessAlreadyEnabled as success rather than a fatal error.
void enable_peer_access(int from, int to) {
  CK(cudaSetDevice(from));
  cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
  if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(e));
    exit(1);
  }
}

// Runs `warmup` untimed copies followed by `iters` timed, stream-serialized
// cudaMemcpyPeerAsync copies of `bytes` from device `src` to device `dst`
// on stream `s`, and returns the total elapsed time in milliseconds for
// the timed iterations.
double timed_copies_ms(int src, void* src_ptr, int dst, void* dst_ptr,
                        size_t bytes, int warmup, int iters, cudaStream_t s) {
  for (int i = 0; i < warmup; ++i)
    CK(cudaMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, s));
  CK(cudaStreamSynchronize(s));

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0, s));
  for (int i = 0; i < iters; ++i)
    CK(cudaMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, s));
  CK(cudaEventRecord(t1, s));
  CK(cudaEventSynchronize(t1));

  float ms = 0.0f;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  CK(cudaEventDestroy(t0));
  CK(cudaEventDestroy(t1));
  return static_cast<double>(ms);
}

double bandwidth_gibps(size_t bytes, int iters, double ms) {
  return static_cast<double>(iters) * bytes / (1u << 30) / (ms / 1e3);
}

double mean_latency_us(int iters, double ms) {
  return ms * 1e3 / iters;
}

}  // namespace

int p2p_probe(int dev_a, int dev_b) {
  int can_ab = 0, can_ba = 0;
  CK(cudaDeviceCanAccessPeer(&can_ab, dev_a, dev_b));
  CK(cudaDeviceCanAccessPeer(&can_ba, dev_b, dev_a));
  printf("P2P %d->%d canAccess=%d\n", dev_a, dev_b, can_ab);
  printf("P2P %d->%d canAccess=%d\n", dev_b, dev_a, can_ba);

  bool p2p_available = can_ab && can_ba;
  if (!p2p_available) {
    printf("P2P: UNAVAILABLE (GeForce driver restriction likely)\n");
    printf(
        "Measuring the staged (host-relayed) transport instead -- a "
        "cross-device copy works without peer access because CUDA stages it "
        "through host memory, and the TP2 all-reduce issues it regardless. "
        "(This probe times cudaMemcpyPeerAsync; the collectives now use the "
        "capture-legal UVA form -- see tools/tp2/transport_probe.cu.)\n");
  } else {
    // Both directions canAccess == 1: enable peer access once in each
    // direction (not twice on the same device -- cudaDeviceEnablePeerAccess
    // returns cudaErrorPeerAccessAlreadyEnabled on a repeat call, which
    // enable_peer_access() tolerates rather than treating as fatal).
    enable_peer_access(dev_a, dev_b);
    enable_peer_access(dev_b, dev_a);
  }

  void *ha = nullptr, *hb = nullptr;
  CK(cudaSetDevice(dev_a));
  CK(cudaMalloc(&ha, kBigBytes));
  CK(cudaSetDevice(dev_b));
  CK(cudaMalloc(&hb, kBigBytes));

  cudaStream_t s;
  CK(cudaStreamCreate(&s));

  const char* label = p2p_available ? "P2P" : "staged";

  double ms_ab = timed_copies_ms(dev_a, ha, dev_b, hb, kBigBytes, kBigWarmup,
                                  kBigTimed, s);
  double bw_ab = bandwidth_gibps(kBigBytes, kBigTimed, ms_ab);
  printf("%s %d->%d one-way bandwidth: %.2f GiB/s over %d copies (256 MiB)\n",
         label, dev_a, dev_b, bw_ab, kBigTimed);

  // The planned all-reduce uses both directions concurrently, so measure
  // the reverse direction too.
  double ms_ba = timed_copies_ms(dev_b, hb, dev_a, ha, kBigBytes, kBigWarmup,
                                  kBigTimed, s);
  double bw_ba = bandwidth_gibps(kBigBytes, kBigTimed, ms_ba);
  printf("%s %d->%d one-way bandwidth: %.2f GiB/s over %d copies (256 MiB)\n",
         label, dev_b, dev_a, bw_ba, kBigTimed);

  // Small-transfer latency dominates the TP2 all-reduce (128 x ~10 KiB per
  // decode token), so measure it regardless of whether P2P is available.
  double ms_small_ab = timed_copies_ms(dev_a, ha, dev_b, hb, kSmallBytes,
                                        kSmallWarmup, kSmallTimed, s);
  double lat_ab_us = mean_latency_us(kSmallTimed, ms_small_ab);
  printf(
      "%s %d->%d small-transfer (10 KiB) latency: %.2f us mean over %d "
      "copies\n",
      label, dev_a, dev_b, lat_ab_us, kSmallTimed);

  double ms_small_ba = timed_copies_ms(dev_b, hb, dev_a, ha, kSmallBytes,
                                        kSmallWarmup, kSmallTimed, s);
  double lat_ba_us = mean_latency_us(kSmallTimed, ms_small_ba);
  printf(
      "%s %d->%d small-transfer (10 KiB) latency: %.2f us mean over %d "
      "copies\n",
      label, dev_b, dev_a, lat_ba_us, kSmallTimed);

  CK(cudaStreamDestroy(s));
  CK(cudaSetDevice(dev_a));
  CK(cudaFree(ha));
  CK(cudaSetDevice(dev_b));
  CK(cudaFree(hb));

  if (!p2p_available) {
    printf(
        "STAGED: canAccess both directions=0, %d->%d=%.2f GiB/s / %.2f us, "
        "%d->%d=%.2f GiB/s / %.2f us (staged transport measured, exit 2)\n",
        dev_a, dev_b, bw_ab, lat_ab_us, dev_b, dev_a, bw_ba, lat_ba_us);
    return 2;
  }

  bool pass = bw_ab >= kMinGiBps && bw_ba >= kMinGiBps;
  printf(
      "%s: canAccess both directions=1, %d->%d=%.2f GiB/s, %d->%d=%.2f "
      "GiB/s (threshold %.1f GiB/s)\n",
      pass ? "PASS" : "FAIL", dev_a, dev_b, bw_ab, dev_b, dev_a, bw_ba,
      kMinGiBps);

  return pass ? 0 : 1;
}

int main(int argc, char** argv) {
  int a = argc > 1 ? atoi(argv[1]) : 0;
  int b = argc > 2 ? atoi(argv[2]) : 1;
  return p2p_probe(a, b);
}
