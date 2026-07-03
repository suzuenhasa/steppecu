// src/device/cuda/block_sink.cuh
//
// The block-spill sink: gets each genome block's finished [P²] f2 + vpair slab
// out of GPU memory to its destination (host RAM or disk) without stalling the
// GPU that is computing the next block. CUDA code — it takes device pointers and
// owns pinned host staging. Tier 0 (Resident) bypasses this seam entirely.
//
// Reference: docs/reference/src_device_cuda_block_sink.cuh.md
#ifndef STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH
#define STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/internal/log.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/cuda/stream.hpp"
#include "steppe/fstats.hpp"

namespace steppe::device {

// Triple-buffer depth — reference §3
inline constexpr int kStreamStagingSlots = 3;

// The block-spill sink interface — reference §4
class BlockSink {
public:
    virtual ~BlockSink() = default;
    virtual void begin(int P, int n_block, const std::vector<int>& block_sizes) = 0;
    virtual void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                             std::size_t slab_elems, cudaStream_t stream) = 0;
    virtual void finish() = 0;
};

// One pinned staging slot — reference §5
struct SinkSlot {
    PinnedBuffer<double> f2;
    PinnedBuffer<double> vpair;
    Event done;
    int block = -1;
};

// Fail-fast slot-drained wait — reference §6
inline void sink_wait_slot_drained(cudaEvent_t done, const char* what) {
    const cudaError_t e = cudaEventSynchronize(done);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + " cudaEventSynchronize: " +
                                 cudaGetErrorString(e));
}

// The shared pinned ring + background writer — reference §7
class StagingRing {
public:
    using DrainFn = std::function<void(SinkSlot& slot)>;

    StagingRing() = default;
    ~StagingRing() { stop_and_join(); teardown_barrier(); }
    StagingRing(const StagingRing&) = delete;
    StagingRing& operator=(const StagingRing&) = delete;
    StagingRing(StagingRing&&) = delete;
    StagingRing& operator=(StagingRing&&) = delete;

    void begin(std::size_t slab_elems, DrainFn drain, const char* what) {
        slab_ = slab_elems;
        slab_bytes_ = slab_ * sizeof(double);
        drain_ = std::move(drain);
        what_ = what;
        if (slab_ == 0) return;
        slots_.resize(kStreamStagingSlots);
        free_.assign(kStreamStagingSlots, true);
        for (SinkSlot& s : slots_) {
            s.f2 = PinnedBuffer<double>(slab_);
            s.vpair = PinnedBuffer<double>(slab_);
            s.block = -1;
        }
        stop_ = false;
        writer_failed_ = false;
        writer_ = std::thread(&StagingRing::writer_loop, this);
    }

    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     cudaStream_t stream) {
        if (slab_ == 0) return;
        const int idx = acquire_slot();
        SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
        s.block = b;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.f2.data(), f2_dev, slab_bytes_,
                                          cudaMemcpyDeviceToHost, stream));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(s.vpair.data(), vpair_dev, slab_bytes_,
                                          cudaMemcpyDeviceToHost, stream));
        STEPPE_CUDA_CHECK(cudaEventRecord(s.done.get(), stream));
        {
            std::lock_guard<std::mutex> lk(mtx_);
            ready_.push(idx);
        }
        cv_work_.notify_one();
    }

    void stop_and_join() {
        if (!writer_.joinable()) return;
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        writer_.join();
    }

    [[nodiscard]] bool writer_failed() const noexcept { return writer_failed_; }
    [[nodiscard]] const std::string& writer_error() const noexcept { return writer_error_; }

private:
    [[nodiscard]] int acquire_slot() {
        std::unique_lock<std::mutex> lk(mtx_);
        int idx = -1;
        cv_free_.wait(lk, [&] {
            for (int i = 0; i < static_cast<int>(free_.size()); ++i)
                if (free_[static_cast<std::size_t>(i)]) { idx = i; return true; }
            return false;
        });
        free_[static_cast<std::size_t>(idx)] = false;
        return idx;
    }

    void writer_loop() {
        for (;;) {
            int idx = -1;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_work_.wait(lk, [&] { return stop_ || !ready_.empty(); });
                if (ready_.empty() && stop_) return;
                idx = ready_.front();
                ready_.pop();
            }
            SinkSlot& s = slots_[static_cast<std::size_t>(idx)];
            try {
                sink_wait_slot_drained(s.done.get(), what_);
                drain_(s);
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(mtx_);
                if (!writer_failed_) { writer_failed_ = true; writer_error_ = ex.what(); }
            }
            {
                std::lock_guard<std::mutex> lk(mtx_);
                free_[static_cast<std::size_t>(idx)] = true;
            }
            cv_free_.notify_one();
        }
    }

    void teardown_barrier() noexcept {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            STEPPE_LOG_WARN("cudaDeviceSynchronize (StagingRing teardown): %s",
                            cudaGetErrorString(e));
    }

    std::size_t slab_ = 0;
    std::size_t slab_bytes_ = 0;
    DrainFn drain_;
    const char* what_ = "";
    std::vector<SinkSlot> slots_;
    std::thread writer_;
    std::mutex mtx_;
    std::condition_variable cv_free_;
    std::condition_variable cv_work_;
    std::queue<int> ready_;
    std::vector<bool> free_;
    bool stop_ = false;
    bool writer_failed_ = false;
    std::string writer_error_;
};

// Tier 1: spill to host RAM — reference §8
class HostRamSink final : public BlockSink {
public:
    explicit HostRamSink(F2BlockTensor& dst) noexcept : host_(dst) {}
    ~HostRamSink() override = default;
    HostRamSink(const HostRamSink&) = delete;
    HostRamSink& operator=(const HostRamSink&) = delete;
    HostRamSink(HostRamSink&&) = delete;
    HostRamSink& operator=(HostRamSink&&) = delete;
    void begin(int P, int n_block, const std::vector<int>& block_sizes) override;
    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     std::size_t slab_elems, cudaStream_t stream) override;
    void finish() override;

private:
    F2BlockTensor& host_;
    int P_ = 0;
    int n_block_ = 0;
    std::size_t slab_ = 0;
    StagingRing ring_;
};

class DiskF2Blocks;

// Tier 2: spill to disk — reference §9
class DiskSink final : public BlockSink {
public:
    explicit DiskSink(std::string path) noexcept : path_(std::move(path)) {}
    ~DiskSink() override;
    DiskSink(const DiskSink&) = delete;
    DiskSink& operator=(const DiskSink&) = delete;
    DiskSink(DiskSink&&) = delete;
    DiskSink& operator=(DiskSink&&) = delete;
    void begin(int P, int n_block, const std::vector<int>& block_sizes) override;
    void spill_block(int b, const double* f2_dev, const double* vpair_dev,
                     std::size_t slab_elems, cudaStream_t stream) override;
    void finish() override;

    void take_descriptor(DiskF2Blocks& out);

private:
    std::string path_;
    int fd_ = -1;
    int P_ = 0;
    int n_block_ = 0;
    std::size_t slab_ = 0;
    std::size_t slab_bytes_ = 0;
    std::uint64_t f2_region_ = 0;
    std::uint64_t vpair_region_ = 0;
    std::vector<int> block_sizes_;
    bool finalized_ = false;
    std::FILE* read_handle_ = nullptr;
    StagingRing ring_;
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_BLOCK_SINK_CUH
