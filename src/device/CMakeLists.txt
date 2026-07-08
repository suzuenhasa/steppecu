# src/device/CMakeLists.txt
#
# Build for steppe_device (alias steppe::device), the one static library that
# holds all of steppe's CUDA. CUDA is linked PRIVATE, so nothing that consumes
# this target compiles against the toolkit — the CUDA-free device/backend.hpp
# seam is all they see.

# The CUDA kernel + backend + CPU-reference source files
add_library(steppe_device STATIC
    cuda/f2_block_kernel.cu
    cuda/f2_batched_kernel.cu
    cuda/qpadm_fit_kernels.cu
    cuda/qpgraph_fit_kernels.cu
    cuda/decode_af_kernel.cu
    cuda/detect_ploidy_kernel.cu
    cuda/transpose_canonical_kernel.cu
    cuda/decode_compact_kernel.cu
    cuda/dstat_kernel.cu
    cuda/fst_wc_kernel.cu
    cuda/cuda_backend_fst.cu
    cuda/fst_allpairs_kernel.cu
    cuda/cuda_backend_fst_allpairs.cu
    cuda/sfs_hist_kernel.cu
    cuda/cuda_backend_sfs.cu
    cuda/pca_standardize_kernel.cu
    cuda/pca_truncated_eig.cu
    cuda/cuda_backend_pca.cu
    cuda/pcangsd_kernels.cu
    cuda/cuda_backend_pcangsd.cu
    cuda/readv2_pack_kernel.cu
    cuda/readv2_mismatch_kernel.cu
    cuda/readv2_bitmatrix.cu
    cuda/cuda_backend_readv2.cu
    cuda/likelihood_tensor.cu
    cuda/cuda_backend_likelihood.cu
    cuda/dates_kernel.cu
    cuda/li_stephens_fb_kernel.cu
    cuda/ancibd_fb_kernel.cu
    cuda/roh_fb_kernel.cu
    cuda/qpfstats_kernel.cu
    cuda/qpfstats_jackknife_kernel.cu
    cuda/ratio_block_jackknife_kernel.cu
    cuda/cuda_backend.cu
    cuda/cuda_backend_qpgraph.cu
    cuda/cuda_backend_dates.cu
    cuda/cuda_backend_li_stephens.cu
    cuda/cuda_backend_ancibd.cu
    cuda/cuda_backend_roh.cu
    cuda/cuda_backend_decode.cu
    cuda/cuda_backend_dstat.cu
    cuda/cuda_backend_qpfstats.cu
    cuda/cuda_backend_fstats_assemble.cu
    cuda/cuda_backend_f2_blocks.cu
    cuda/cuda_backend_qpadm_fit.cu
    cuda/device_partial.cu
    cuda/device_f2_blocks.cu
    cuda/device_decode_result.cu
    cuda/p2p_combine.cu
    cuda/block_sink.cu
    cuda/f2_blocks_out.cu
    cpu/cpu_backend.cpp
    host_ram.cpp
    resources.cpp
    shard_plan.cpp)
add_library(steppe::device ALIAS steppe_device)

# Include path: the src/ root, PUBLIC for the device/backend.hpp seam
target_include_directories(steppe_device PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Link libraries; CUDA linked PRIVATE seals the CUDA-free layering
target_link_libraries(steppe_device
    PUBLIC  steppe::api steppe::core_internal
    PRIVATE steppe::warnings CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)

target_compile_features(steppe_device PUBLIC cxx_std_20)

# STEPPE_HAVE_EMU_TUNING — the fixed-slice Ozaki engagement gate
if(STEPPE_HAVE_EMU_TUNING)
  target_compile_definitions(steppe_device PRIVATE STEPPE_HAVE_EMU_TUNING=1)
endif()

# STEPPE_NVTX — the profiler phase-range gate
if(STEPPE_NVTX)
  target_compile_definitions(steppe_device PRIVATE STEPPE_NVTX)
endif()

# Separable compilation + PIC (cross-TU kernel launch, device-LTO)
set_target_properties(steppe_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)

# Optional fatbin compression, Release-only opt-in
if(STEPPE_COMPRESS_FATBIN)
  target_compile_options(steppe_device PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--compress-mode=size>)
endif()
