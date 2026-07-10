// src/device/wave_budget.hpp
//
// Sizes the batch-over-items WAVE width W against BOTH the live free VRAM and the
// free pinned-host RAM, so a map-only wave sweep (roh, li_stephens) reuses one set
// of W-wide staging buffers and never exceeds either budget. CUDA-free, host-pure
// std::size_t math (mirrors decode_budget.hpp), so it is unit-testable with no GPU
// and includable from the CUDA-free layer.
//
// The DUAL budget is the point: a W sized to fill 32-96 GB of VRAM can exhaust the
// smaller pinned-host allocation (WaveStage allocates pinned host per item too), so
// wave_width caps W by min(vram-derived, host-derived). This closes the latent
// pinned-host-RAM budget bug that the private roh wave loop (which sized W off VRAM
// alone) would otherwise carry.
//
// Reference: docs/planning/batch-over-items-driver-design.md §(b)
#ifndef STEPPE_DEVICE_WAVE_BUDGET_HPP
#define STEPPE_DEVICE_WAVE_BUDGET_HPP

#include <cstddef>
#include <cstdlib>

namespace steppe::device {

// Named constants — mirror the roh wave loop's inline 90%/1 GB choice exactly, so
// the VRAM-derived W is byte-identical to the pre-extraction hand-sum.
inline constexpr double      kWaveVramFraction = 0.9;
inline constexpr std::size_t kWaveReserveBytes = std::size_t{1} << 30;  // 1 GB headroom

// PerItemBytes — declarative footprint builder. Kills roh's error-prone 12-term
// hand-sum (cuda_backend_roh.cu:232): the consumer enumerates its own per-item
// buffers with dev_add<T>/host_add<T>, and the accumulated .dev/.host feed
// wave_width. Device-only scratch adds to .dev only; a staging pair (pinned<->device)
// adds to BOTH.
struct PerItemBytes {
    std::size_t dev = 0;
    std::size_t host = 0;
    template <class T>
    PerItemBytes& dev_add(std::size_t elems) noexcept {
        dev += elems * sizeof(T);
        return *this;
    }
    template <class T>
    PerItemBytes& host_add(std::size_t elems) noexcept {
        host += elems * sizeof(T);
        return *this;
    }
    // Convenience for a staging pair whose pinned + device halves are the same T
    // and element count (the common WaveStage<T> case).
    template <class T>
    PerItemBytes& pair_add(std::size_t elems) noexcept {
        dev_add<T>(elems);
        host_add<T>(elems);
        return *this;
    }
};

namespace detail {

// Largest count of items whose per_item footprint fits one budgeted axis. Replicates
// the roh loop's math term-for-term: reserve headroom (or halve if free <= reserve),
// take `fraction` of the remainder, integer-divide by per_item. A zero per_item means
// this axis imposes no bound (return a sentinel that min() will discard).
[[nodiscard]] inline std::size_t wave_axis_cap(std::size_t free_bytes,
                                               std::size_t per_item_bytes, double fraction,
                                               std::size_t reserve) noexcept {
    if (per_item_bytes == 0) return static_cast<std::size_t>(-1);  // unbounded on this axis
    std::size_t budget = (free_bytes > reserve) ? (free_bytes - reserve) : (free_bytes / 2);
    budget = static_cast<std::size_t>(static_cast<double>(budget) * fraction);
    return budget / per_item_bytes;
}

}  // namespace detail

// Largest wave width W in [1, N] whose per-item footprint fits BOTH budgets.
//   vram_free / host_free: measured by the CALLER, AFTER the shared resident operand
//     is up (so the VRAM measurement already reflects it — the decode_budget seam).
//   per_item_dev_bytes / per_item_host_bytes: the caller's own PerItemBytes sums; the
//     sizer never infers layout.
// W = clamp( min(vram_cap, host_cap), 1, N ), unless STEPPE_WAVE_WIDTH forces it.
// A positive env value pins W (still clamped to [1, N]); unset/invalid uses the
// dual-budget math. With per_item_host_bytes == 0 (no pinned staging) this reduces to
// the VRAM-only cap, byte-identical to the pre-extraction roh W.
[[nodiscard]] inline long wave_width(std::size_t vram_free, std::size_t per_item_dev_bytes,
                                     std::size_t host_free, std::size_t per_item_host_bytes, long N,
                                     const char* env_var = "STEPPE_WAVE_WIDTH",
                                     double fraction = kWaveVramFraction,
                                     std::size_t reserve = kWaveReserveBytes) noexcept {
    if (N <= 0) return 0;
    // Env override: a positive STEPPE_WAVE_WIDTH pins W (still clamped below).
    if (env_var != nullptr) {
        if (const char* e = std::getenv(env_var)) {
            char* end = nullptr;
            const long forced = std::strtol(e, &end, 10);
            if (end != e && forced > 0) {
                return (forced > N) ? N : forced;
            }
        }
    }
    const std::size_t vcap = detail::wave_axis_cap(vram_free, per_item_dev_bytes, fraction, reserve);
    const std::size_t hcap = detail::wave_axis_cap(host_free, per_item_host_bytes, fraction, reserve);
    std::size_t w = (vcap < hcap) ? vcap : hcap;  // cap by the binding axis
    if (w < 1u) w = 1u;
    if (w > static_cast<std::size_t>(N)) w = static_cast<std::size_t>(N);
    return static_cast<long>(w);
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_WAVE_BUDGET_HPP
