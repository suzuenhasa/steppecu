// src/device/cuda/pinned_buffer.cuh
//
// PinnedBuffer<T> + RegisteredHostRegion — the move-only RAII owners of
// PAGE-LOCKED (pinned) host memory (architecture.md §2 RAII, §7, §11.1; ROADMAP
// §5). This is one of the THREE allowlisted translation units permitted to call
// the host-side allocation family directly (architecture.md §2 DRY grep gate:
// `device_buffer.cuh`, `allocator.cu`, `pinned_buffer.cuh`) — here the
// `cudaHostAlloc`/`cudaFreeHost` (PinnedBuffer) and `cudaHostRegister`/
// `cudaHostUnregister` (RegisteredHostRegion) calls; all other code takes
// non-owning views and never touches the allocation family.
//
// WHY PINNED (M4.5 SPMG overlap, perf-discovery.md P4 / L2 / W9). `cudaMemcpyAsync`
// is asynchronous with respect to the host ONLY when the host memory is
// page-locked; from PAGEABLE host memory it falls back to a synchronous,
// host-blocking staging copy and the supposedly-async H2D/D2H actually BLOCKS the
// issuing thread (CUDA C Programming Guide §3.2.8.4 "Asynchronous Concurrent
// Execution / Overlap of Data Transfer and Kernel Execution"; CUDA Runtime API,
// `cudaMemcpyAsync`: "For transfers from pageable host memory ... the function is
// synchronous"; §3.2.5 "Page-Locked Host Memory"). Under the §11.4 multi-GPU
// fan-out that blocking is the measured ~44 % pageable `cudaMemcpyAsync`
// (perf-discovery §2) that prevents device A's H2D from overlapping device B's
// compute. Pinning the staging host memory makes the copy a genuine DMA that
// proceeds concurrently with kernels on the per-device non-blocking stream (P2).
//
// PIN ONLY THE STAGING SLOTS (architecture.md §11.1). Page-locked memory is a
// scarce kernel resource (it is non-pageable physical RAM, bounded by
// RLIMIT_MEMLOCK); over-pinning degrades the whole system (CUDA C Programming
// Guide §3.2.5). So we pin only the buffers actually fed to `cudaMemcpyAsync`, and
// — critically — pinning is a PERF optimization, never a correctness precondition:
// a failed pin (e.g. RLIMIT_MEMLOCK exhausted) DEGRADES GRACEFULLY to the pageable
// path with a debug warning, it does NOT throw (the copy still completes, just
// synchronously). Pinned vs pageable moves the IDENTICAL bytes, so this is
// PARITY-NEUTRAL (architecture.md §11.4, §12 — a data-movement lever only).
//
// This is a CUDA header: PRIVATE to steppe_device (architecture.md §4).
#ifndef STEPPE_DEVICE_CUDA_PINNED_BUFFER_CUH
#define STEPPE_DEVICE_CUDA_PINNED_BUFFER_CUH

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <limits>           // std::numeric_limits — the n*sizeof(T) overflow guard
#include <source_location>  // std::source_location — call site for the typed throw
#include <utility>

#include "core/internal/host_device.hpp"  // STEPPE_ASSERT (the one debug-only fail-fast)
#include "core/internal/log.hpp"  // STEPPE_LOG_WARN (the one teardown-warning sink)
#include "device/cuda/check.cuh"  // STEPPE_CUDA_CHECK, STEPPE_CUDA_WARN, CudaError

namespace steppe::device {

/// Owning, move-only PAGE-LOCKED host allocation of `n` elements of `T`
/// (`cudaHostAlloc`/`cudaFreeHost`). Use as a reusable staging slot the host
/// fills (or the device D2H fills) and `cudaMemcpyAsync` transfers — the classic
/// double-buffered pinned pipeline (architecture.md §11.1). Hands out a raw host
/// pointer via `data()`; never copies. Follows the §7 RAII shape exactly
/// (move-construct + move-assign + dtor->destroy()-with-debug-log, never throws).
template <class T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    /// Allocate `n` page-locked elements. `n == 0` is a no-op (null pointer, zero
    /// size) — not an error. FAIL-FAST on a size overflow, mirroring
    /// DeviceBuffer: the byte request `n * sizeof(T)` is a `std::size_t` product
    /// whose unsigned overflow is *defined* (modular), so an unchecked multiply
    /// would silently WRAP and under-allocate; reject any `n` whose byte product
    /// would exceed `SIZE_MAX` with a typed `CudaError` (architecture.md §2;
    /// cleanup B23). `cudaHostAlloc` with `cudaHostAllocDefault` yields ordinary
    /// page-locked memory usable by `cudaMemcpyAsync` on any stream (CUDA Runtime
    /// API, `cudaHostAlloc`).
    explicit PinnedBuffer(std::size_t n) : size_(n) {
        if (n) {
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw CudaError(cudaErrorMemoryAllocation,
                                "PinnedBuffer: n * sizeof(T) overflows size_t",
                                std::source_location::current());
            }
            STEPPE_CUDA_CHECK(  // ALLOWLISTED TU
                cudaHostAlloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T),
                              cudaHostAllocDefault));
        }
    }

    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}

    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = std::exchange(o.ptr_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() { reset(); }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }

private:
    void reset() noexcept {
        if (ptr_) {
            // Destructor never throws (architecture.md §7); a nonzero teardown
            // status is reported to the debug-only warning sink, never thrown.
            const cudaError_t e = cudaFreeHost(ptr_);  // ALLOWLISTED TU
            if (e != cudaSuccess) {
                STEPPE_LOG_WARN("cudaFreeHost at PinnedBuffer teardown: %s",
                                cudaGetErrorString(e));
            }
        }
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

/// Move-only RAII guard that PAGE-LOCKS an EXISTING host buffer IN PLACE via
/// `cudaHostRegister`, and `cudaHostUnregister`s it on scope exit
/// (architecture.md §7, §11.1). This is the zero-copy alternative to PinnedBuffer
/// for buffers the backend does NOT own — the caller's Q/V/N H2D source views and
/// the result D2H destination vectors: registering the caller's pages turns the
/// EXISTING `cudaMemcpyAsync` into a true async DMA WITHOUT an extra host copy
/// (CUDA Runtime API, `cudaHostRegister`: "Page-locks the memory range ... and
/// maps it for the device(s)"; the range need not be page-aligned — the driver
/// rounds to host page boundaries).
///
/// GRACEFUL DEGRADE (perf-discovery.md P4): pinning is a PERF lever, never a
/// correctness precondition. A failed `cudaHostRegister` — RLIMIT_MEMLOCK
/// exhausted (`cudaErrorMemoryAllocation`), an already-registered/overlapping
/// range (`cudaErrorHostMemoryAlreadyRegistered`), or any other status — routes
/// through the NON-throwing STEPPE_CUDA_WARN and leaves `registered_ == false`:
/// the subsequent `cudaMemcpyAsync` still runs correctly, just synchronously over
/// pageable memory (the pre-P4 behavior). We NEVER throw out of the pin attempt,
/// so a low MEMLOCK ulimit cannot crash the run (it only forfeits the overlap).
/// The sticky last-error a failed register may set is CLEARED so a later
/// post-launch `cudaGetLastError` does not misattribute it.
///
/// CONSTNESS: a const `T*` source (an H2D upload reads from it) is registered via
/// a `const_cast` to `void*` — `cudaHostRegister` does not modify the bytes, it
/// only changes the page state, so registering a logically-const source is sound
/// (the API takes `void*` for both read and write ranges).
class RegisteredHostRegion {
public:
    RegisteredHostRegion() = default;

    /// Page-lock `[ptr, ptr+bytes)` in place. A null `ptr` or `bytes == 0` is a
    /// no-op (nothing to pin). On any non-success status the region is left
    /// UNREGISTERED (graceful pageable fallback) — the ctor never throws.
    RegisteredHostRegion(const void* ptr, std::size_t bytes) {
        if (ptr == nullptr || bytes == 0) return;
        // cudaHostRegister takes void* (it page-locks, it does not write the
        // bytes), so a logically-const H2D source registers soundly via const_cast.
        void* p = const_cast<void*>(ptr);
        const cudaError_t s =  // ALLOWLISTED TU (non-throwing: graceful degrade)
            STEPPE_CUDA_WARN(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
        if (s == cudaSuccess) {
            ptr_ = p;  // own the unregister
        } else {
            // Discard the sticky last-error the failed register set, so a later
            // post-launch cudaGetLastError does not misattribute this tolerated
            // degrade as a kernel-launch failure (CUDA Runtime API: cudaGetLastError
            // reads AND resets the sticky error). The H2D/D2H below proceeds over
            // pageable memory — correct, just synchronous.
            (void)cudaGetLastError();
        }
    }

    RegisteredHostRegion(RegisteredHostRegion&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)) {}

    RegisteredHostRegion& operator=(RegisteredHostRegion&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = std::exchange(o.ptr_, nullptr);
        }
        return *this;
    }

    RegisteredHostRegion(const RegisteredHostRegion&) = delete;
    RegisteredHostRegion& operator=(const RegisteredHostRegion&) = delete;

    ~RegisteredHostRegion() { reset(); }

    /// Did the pin actually take effect? (false ⇒ pageable fallback was used.)
    [[nodiscard]] bool registered() const noexcept { return ptr_ != nullptr; }

private:
    void reset() noexcept {
        if (ptr_) {
            // Destructor never throws (architecture.md §7); a nonzero unregister
            // status is reported to the debug-only warning sink, never thrown.
            const cudaError_t e = cudaHostUnregister(ptr_);  // ALLOWLISTED TU
            if (e != cudaSuccess) {
                STEPPE_LOG_WARN("cudaHostUnregister at teardown: %s",
                                cudaGetErrorString(e));
            }
        }
        ptr_ = nullptr;
    }

    void* ptr_ = nullptr;  // non-null ⇔ we registered and must unregister
};

/// A small fixed-capacity, move-only cache of PERSISTENT host-page registrations
/// (architecture.md §7, §11.1). It exists to AMORTIZE the `cudaHostRegister` cost,
/// which is the whole point of P4: registering a multi-GB host range is itself a
/// heavyweight, page-walking, ~50–360 ms operation (MEASURED on rtxbox), so pinning
/// the SAME caller buffer on EVERY `compute_f2_blocks` call is a net LOSS — the
/// per-call register tax dwarfs the overlap win. The win only materializes when the
/// registration is paid ONCE and reused: under the §11.4 two-device fan-out, two
/// CONCURRENT pinned H2D DMAs run at ~full PCIe bandwidth each (MEASURED ~51 ms/iter
/// per device), whereas two concurrent PAGEABLE H2Ds contend and serialize on the
/// driver's internal staging (~109 ms/iter) — a measured ~2× per-device copy
/// speedup, but ONLY once the register cost is amortized across the repeated calls.
///
/// USAGE (the backend's H2D Q/V/N sources): call `ensure(ptr, bytes)` before each
/// `cudaMemcpyAsync(dst, ptr, bytes)`. The FIRST time a `(ptr, bytes)` is seen it is
/// registered and the registration is RETAINED; subsequent calls with the same range
/// are a cheap no-op (already pinned) — so a workload that reuses the same host Q/V/N
/// across iterations (the bench; the M5 streaming tile reuse) pays the register cost
/// once. A range not already cached EVICTS the round-robin slot (unregistering
/// whatever it held) and registers the new range — bounding the resident pinned set
/// to `kSlots` ranges (pin only the staging slots, architecture.md §11.1). With
/// `kSlots` == the 3 Q/V/N inputs, the steady state never self-evicts. All
/// registrations unregister at cache teardown (RAII).
///
/// WHY NOT the D2H result destinations: those are FRESHLY allocated `std::vector`s on
/// every call (a new base pointer each time), so caching never hits — they would pay
/// the register tax every call with zero amortization, a strict loss. The backend
/// therefore leaves the D2H result copies PAGEABLE and caches only the stable H2D
/// inputs.
///
/// GRACEFUL DEGRADE + PARITY: `ensure` delegates to RegisteredHostRegion, so a failed
/// pin (RLIMIT_MEMLOCK) degrades to the pageable path with a debug warning — never a
/// crash. Pinned vs pageable moves the identical bytes (architecture.md §11.4, §12) —
/// PARITY-NEUTRAL.
class PinnedRegistryCache {
public:
    PinnedRegistryCache() = default;

    PinnedRegistryCache(PinnedRegistryCache&&) noexcept = default;
    PinnedRegistryCache& operator=(PinnedRegistryCache&&) noexcept = default;
    PinnedRegistryCache(const PinnedRegistryCache&) = delete;
    PinnedRegistryCache& operator=(const PinnedRegistryCache&) = delete;
    ~PinnedRegistryCache() = default;  // RegisteredHostRegion dtors unregister all

    /// Ensure `[ptr, ptr+bytes)` is page-locked, reusing an existing registration of
    /// the SAME (ptr, bytes) if present (the amortized fast path). A null/zero range
    /// is a no-op. If the range is not cached, evict the round-robin slot (its
    /// RegisteredHostRegion dtor unregisters the old range) and register the new one.
    ///
    /// PRECONDITION — EVICTION vs IN-FLIGHT COPY (14.4): on a cache MISS the victim
    /// round-robin slot is evicted, whose RegisteredHostRegion move-assign runs the OLD
    /// region's dtor → `cudaHostUnregister` SYNCHRONOUSLY with NO stream sync. The CUDA
    /// 13.x Runtime API requires that no `cudaMemcpyAsync` be in flight over a range when
    /// it is unregistered — `cudaHostUnregister` unmaps the range and reverts it to
    /// PAGEABLE, so an in-flight DMA would then read pageable memory (a use-after-unregister
    /// race, the exact unsafe overlap this cache exists to prevent). The CALLER must
    /// therefore NOT cause a range with an outstanding async copy to be evicted (it must
    /// have synced the stream / gated reuse via cudaEvent before any miss could displace
    /// that range). `ensure` deliberately does NOT add a hidden sync here — that would
    /// serialize the hot H2D path. With `kSlots == 3 == ` the Q/V/N inputs this never fires
    /// in steady state; the only way to reach it is staging a 4th distinct stable H2D range,
    /// which the debug assert below makes LOUD (a real registered victim about to be
    /// unregistered) so silent self-eviction cannot reach the unregister while a copy is live.
    void ensure(const void* ptr, std::size_t bytes) {
        if (ptr == nullptr || bytes == 0) return;
        for (const Slot& s : slots_) {
            if (s.ptr == ptr && s.bytes == bytes && s.reg.registered()) return;
        }
        // Miss: register into the round-robin slot (evicting whatever it held). The
        // assignment runs the OLD RegisteredHostRegion's dtor first (unregisters the
        // evicted range), then move-assigns the freshly-registered region in.
        Slot& dst = slots_[next_];
        // LOUD SELF-EVICTION GUARD (14.4): in the documented steady state (3 inputs ==
        // kSlots) the victim slot is empty, so eviction never unregisters a live range. A
        // miss whose victim is ACTUALLY registered means a 4th distinct stable range is
        // being staged — the eviction-vs-in-flight-copy precondition above is now in play
        // and the silent unregister could race a DMA. Fail loudly under debug /
        // compute-sanitizer rather than silently revert a possibly-in-flight range to
        // pageable; release is unaffected (STEPPE_ASSERT compiles out, the bounded
        // round-robin eviction still degrades gracefully).
        STEPPE_ASSERT(!dst.reg.registered(),
                      "PinnedRegistryCache::ensure: evicting a still-registered slot "
                      "(a 4th distinct H2D range exceeds kSlots); the evicted range must "
                      "have no outstanding async copy (no hidden sync is performed here)");
        dst.reg = RegisteredHostRegion(ptr, bytes);
        dst.ptr = dst.reg.registered() ? ptr : nullptr;  // only remember a real pin
        dst.bytes = dst.reg.registered() ? bytes : 0;
        next_ = (next_ + 1) % kSlots;
    }

private:
    // Three slots: exactly the Q/V/N H2D inputs the f2 path stages per call, so all
    // three stay resident together across iterations (no self-eviction). A 4th input
    // would evict the oldest — a bounded pinned footprint, never an unbounded pin.
    static constexpr std::size_t kSlots = 3;
    struct Slot {
        const void* ptr = nullptr;
        std::size_t bytes = 0;
        RegisteredHostRegion reg{};
    };
    std::array<Slot, kSlots> slots_{};
    std::size_t next_ = 0;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_PINNED_BUFFER_CUH
