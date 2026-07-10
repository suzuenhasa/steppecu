// src/device/cuda/wave_batch.cuh
//
// The map-only, VRAM+host-budgeted WAVE core: a reusable pinned<->device staging
// pair (WaveStage<T>) plus the ordered-consume loop (wave_map). Extracted from the
// private roh wave loop (cuda_backend_roh.cu:258-315) so roh and li_stephens share
// one parity-safe sweep instead of each re-deriving it.
//
// This is MAP-ONLY: each item is independent, its outputs are a disjoint per-item
// slice, and the driver owns only the sequencing + the strict-ascending consume
// contract. No reduction / accumulator / multi-pass machinery (see the design doc's
// non-goals) — a reduce-then-map path is deliberately NOT expressed here.
//
// Reference: docs/planning/batch-over-items-driver-design.md §(b)
#ifndef STEPPE_DEVICE_CUDA_WAVE_BATCH_CUH
#define STEPPE_DEVICE_CUDA_WAVE_BATCH_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>

#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/pinned_buffer.cuh"

namespace steppe::device {

// WaveStage<T> — one reusable pinned<->device staging pair, sized ONCE to
// W*elems_per_item and reused every wave. Owns both halves (RAII, move-only). Lives
// on the consumer's stack, declared before wave_map. `host()` is pinned staging the
// build callback fills / the consume callback reads; `device()` is the operand the
// kernel/GEMM reads or writes. h2d/d2h move only the LIVE prefix [0, count*epi).
//
// An input stage: build fills host()[0, Wc*epi) then h2d(Wc); the kernel reads
// device(). An output stage: the kernel writes device(); launch d2h(Wc); consume
// reads host()[0, Wc*epi) after wave_map's per-wave sync.
template <class T>
class WaveStage {
public:
    WaveStage() = default;

    // W items, elems_per_item elements each. Allocates W*elems_per_item on both the
    // pinned-host and device sides. The stream argument is accepted for symmetry with
    // the copy calls (allocation itself is synchronous), so the ctor documents which
    // stream the stage will be driven on.
    WaveStage(long W, std::size_t elems_per_item, cudaStream_t /*s*/)
        : host_(static_cast<std::size_t>(W < 0 ? 0 : W) * elems_per_item),
          dev_(static_cast<std::size_t>(W < 0 ? 0 : W) * elems_per_item),
          epi_(elems_per_item) {}

    WaveStage(WaveStage&&) noexcept = default;
    WaveStage& operator=(WaveStage&&) noexcept = default;
    WaveStage(const WaveStage&) = delete;
    WaveStage& operator=(const WaveStage&) = delete;
    ~WaveStage() = default;

    [[nodiscard]] T* host() noexcept { return host_.data(); }
    [[nodiscard]] const T* host() const noexcept { return host_.data(); }
    [[nodiscard]] T* device() noexcept { return dev_.data(); }
    [[nodiscard]] const T* device() const noexcept { return dev_.data(); }
    [[nodiscard]] std::size_t elems_per_item() const noexcept { return epi_; }

    // Upload the live prefix: count items * epi elements, host -> device.
    void h2d(long count, cudaStream_t s) {
        if (count <= 0 || epi_ == 0) return;
        h2d_async(dev_, host_.data(), static_cast<std::size_t>(count) * epi_, s);
    }
    // Download the live prefix: count items * epi elements, device -> host.
    void d2h(long count, cudaStream_t s) {
        if (count <= 0 || epi_ == 0) return;
        d2h_async(host_.data(), dev_, static_cast<std::size_t>(count) * epi_, s);
    }

private:
    PinnedBuffer<T> host_;
    DeviceBuffer<T> dev_;
    std::size_t epi_ = 0;
};

// WaveMapOps — the three per-wave callbacks. Each receives (w0, Wc): the wave's
// absolute item offset and its live count. The consumer captures its own WaveStages
// and the once-resident shared operand.
//   build(w0, Wc):   fill each item's DISJOINT pinned slice (may host-thread) and
//                    h2d the live [0, Wc) prefix of every input stage.
//   launch(w0, Wc):  one grid of Wc item-blocks against the resident operand, then
//                    d2h the output stage(s).
//   consume(w0, Wc): invoked by wave_map in STRICT ascending w0, AFTER the per-wave
//                    stream sync, so the emitted table is byte-stable regardless of
//                    GPU completion order.
struct WaveMapOps {
    std::function<void(long w0, long Wc)> build;
    std::function<void(long w0, long Wc)> launch;
    std::function<void(long w0, long Wc)> consume;
};

// wave_map — the map-only sweep with ORDERED CONSUME AS A CONTRACT. Generalizes
// roh.cu:258-315: for each wave [w0, w0+Wc), build -> launch -> sync(s) -> consume,
// with consume guaranteed strictly ascending. The shared operand is uploaded ONCE by
// the caller before wave_map and is const for the run; the staging buffers are sized
// to W and reused every wave. No overlap / double-buffering (sync-per-wave, roh's
// deliberate model).
inline void wave_map(long N, long W, const WaveMapOps& ops, cudaStream_t s) {
    if (N <= 0 || W <= 0) return;
    for (long w0 = 0; w0 < N; w0 += W) {
        const long Wc = std::min<long>(W, N - w0);  // items in this wave
        if (ops.build) ops.build(w0, Wc);
        if (ops.launch) ops.launch(w0, Wc);
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(s));  // input H2D + kernel + output D2H complete
        if (ops.consume) ops.consume(w0, Wc);         // strict ascending w0
    }
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_WAVE_BATCH_CUH
