# Code Review Index

This directory contains per-file senior-developer reviews for the `steppe` C++/CUDA codebase, plus cross-cutting and synthesized reports.

**Scope:** `.cpp`, `.cu`, `.cuh`, `.hpp` files only. Python bindings and tests were excluded per request.  
**Style:** Each review follows the tone and structure of `docs/kimiexample.md` — what's good, what a senior dev would flag, the slop test, characterization, and a verdict.

---

## Syntheses and cross-cutting reports

- [`synthesis_wave_1.md`](synthesis_wave_1.md) — Synthesis of wave 1: core CUDA kernels, multigpu orchestration, emit/config layer, and public headers.
- [`synthesis_wave_2.md`](synthesis_wave_2.md) — Synthesis of wave 2: host-side algorithms, I/O readers, Python binding, and entry point.
- [`cross_cutting_output_layer_review.md`](cross_cutting_output_layer_review.md) — Dedicated cross-cutting review of the CLI output/emit layer, confirming the duplicated CSV/emit suspicion.

---

## Per-file reviews by category

### CUDA kernels and device code (`src/device/cuda/`)

- [`src_device_cuda_block_sink.md`](src_device_cuda_block_sink.md)
- [`src_device_cuda_cuda_backend.md`](src_device_cuda_cuda_backend.md)
- [`src_device_cuda_dates_kernel.md`](src_device_cuda_dates_kernel.md)
- [`src_device_cuda_decode_af_kernel.md`](src_device_cuda_decode_af_kernel.md)
- [`src_device_cuda_decode_compact_kernel.md`](src_device_cuda_decode_compact_kernel.md)
- [`src_device_cuda_detect_ploidy_kernel.md`](src_device_cuda_detect_ploidy_kernel.md)
- [`src_device_cuda_device_decode_result.md`](src_device_cuda_device_decode_result.md)
- [`src_device_cuda_device_f2_blocks.md`](src_device_cuda_device_f2_blocks.md)
- [`src_device_cuda_device_partial.md`](src_device_cuda_device_partial.md)
- [`src_device_cuda_dstat_kernel.md`](src_device_cuda_dstat_kernel.md)
- [`src_device_cuda_f2_block_kernel.md`](src_device_cuda_f2_block_kernel.md)
- [`src_device_cuda_f2_blocks_kernel.md`](src_device_cuda_f2_blocks_kernel.md)
- [`src_device_cuda_f2_blocks_out.md`](src_device_cuda_f2_blocks_out.md)
- [`src_device_cuda_handles.md`](src_device_cuda_handles.md)
- [`src_device_cuda_p2p_combine.md`](src_device_cuda_p2p_combine.md)
- [`src_device_cuda_qpadm_fit_kernels.md`](src_device_cuda_qpadm_fit_kernels.md)
- [`src_device_cuda_qpfstats_jackknife_kernel.md`](src_device_cuda_qpfstats_jackknife_kernel.md)
- [`src_device_cuda_qpfstats_kernel.md`](src_device_cuda_qpfstats_kernel.md)
- [`src_device_cuda_qpgraph_fit_kernels.md`](src_device_cuda_qpgraph_fit_kernels.md)
- [`src_device_cuda_ratio_block_jackknife_kernel.md`](src_device_cuda_ratio_block_jackknife_kernel.md)
- [`src_device_cuda_transpose_canonical_kernel.md`](src_device_cuda_transpose_canonical_kernel.md)

### Device abstraction and orchestration (`src/device/`)

- [`src_device_backend.md`](src_device_backend.md)
- [`src_device_backend_factory.md`](src_device_backend_factory.md)
- [`src_device_cpu_cpu_backend.md`](src_device_cpu_cpu_backend.md)
- [`src_device_host_ram.md`](src_device_host_ram.md)
- [`src_device_resources.md`](src_device_resources.md)
- [`src_device_shard_plan.md`](src_device_shard_plan.md)

### Core qpadm / qpgraph / f-stat algorithms (`src/core/qpadm/`)

- [`src_core_qpadm_f3.md`](src_core_qpadm_f3.md)
- [`src_core_qpadm_f4.md`](src_core_qpadm_f4.md)
- [`src_core_qpadm_f4ratio.md`](src_core_qpadm_f4ratio.md)
- [`src_core_qpadm_model_search.md`](src_core_qpadm_model_search.md)
- [`src_core_qpadm_model_search_core.md`](src_core_qpadm_model_search_core.md)
- [`src_core_qpadm_nested_models.md`](src_core_qpadm_nested_models.md)
- [`src_core_qpadm_qpadm_fit.md`](src_core_qpadm_qpadm_fit.md)
- [`src_core_qpadm_qpgraph_enumerate.md`](src_core_qpadm_qpgraph_enumerate.md)
- [`src_core_qpadm_qpgraph_fit.md`](src_core_qpadm_qpgraph_fit.md)
- [`src_core_qpadm_qpgraph_model.md`](src_core_qpadm_qpgraph_model.md)
- [`src_core_qpadm_qpgraph_search.md`](src_core_qpadm_qpgraph_search.md)

### Core statistics (`src/core/stats/`)

- [`src_core_stats_dates.md`](src_core_stats_dates.md)
- [`src_core_stats_dstat.md`](src_core_stats_dstat.md)
- [`src_core_stats_qpfstats.md`](src_core_stats_qpfstats.md)

### Core f-stats and multigpu (`src/core/fstats/`)

- [`src_core_fstats_f2_blocks_multigpu.md`](src_core_fstats_f2_blocks_multigpu.md)
- [`src_core_fstats_f2_blocks_multigpu_core.md`](src_core_fstats_f2_blocks_multigpu_core.md)
- [`src_core_fstats_f2_combine.md`](src_core_fstats_f2_combine.md)
- [`src_core_fstats_f2_from_blocks.md`](src_core_fstats_f2_from_blocks.md)

### Internal headers (`src/core/internal/`)

- [`src_core_internal_host_device.md`](src_core_internal_host_device.md)
- [`src_core_internal_launch_config.md`](src_core_internal_launch_config.md)
- [`src_core_internal_log.md`](src_core_internal_log.md)
- [`src_core_internal_small_linalg.md`](src_core_internal_small_linalg.md)
- [`src_core_internal_views.md`](src_core_internal_views.md)

### Configuration (`src/core/config/`)

- [`src_core_config_build_result.md`](src_core_config_build_result.md)
- [`src_core_config_cli_args.md`](src_core_config_cli_args.md)
- [`src_core_config_config_builder.md`](src_core_config_config_builder.md)
- [`src_core_config_run_config.md`](src_core_config_run_config.md)

### I/O readers and filters (`src/io/`)

- [`src_io_eigenstrat_format.md`](src_io_eigenstrat_format.md)
- [`src_io_filter_include_exclude.md`](src_io_filter_include_exclude.md)
- [`src_io_filter_mind_prepass.md`](src_io_filter_mind_prepass.md)
- [`src_io_filter_snp_filter.md`](src_io_filter_snp_filter.md)
- [`src_io_geno_reader.md`](src_io_geno_reader.md)
- [`src_io_genotype_source.md`](src_io_genotype_source.md)
- [`src_io_ind_reader.md`](src_io_ind_reader.md)
- [`src_io_plink_reader.md`](src_io_plink_reader.md)
- [`src_io_ploidy_detect.md`](src_io_ploidy_detect.md)
- [`src_io_snp_reader.md`](src_io_snp_reader.md)

### CLI application layer (`src/app/`)

- [`src_app_cli_parse.md`](src_app_cli_parse.md)
- [`src_app_cmd_dates.md`](src_app_cmd_dates.md)
- [`src_app_cmd_extract_f2.md`](src_app_cmd_extract_f2.md)
- [`src_app_cmd_f3.md`](src_app_cmd_f3.md)
- [`src_app_cmd_f4.md`](src_app_cmd_f4.md)
- [`src_app_cmd_f4ratio.md`](src_app_cmd_f4ratio.md)
- [`src_app_cmd_fstat_sweep.md`](src_app_cmd_fstat_sweep.md)
- [`src_app_cmd_qpadm.md`](src_app_cmd_qpadm.md)
- [`src_app_cmd_qpdstat.md`](src_app_cmd_qpdstat.md)
- [`src_app_cmd_qpfstats.md`](src_app_cmd_qpfstats.md)
- [`src_app_cmd_qpgraph.md`](src_app_cmd_qpgraph.md)
- [`src_app_cmd_qpwave.md`](src_app_cmd_qpwave.md)
- [`src_app_cmd_rotate.md`](src_app_cmd_rotate.md)
- [`src_app_extract_f2_core.md`](src_app_extract_f2_core.md)
- [`src_app_f2_dir_io.md`](src_app_f2_dir_io.md)
- [`src_app_f2_dir_writer.md`](src_app_f2_dir_writer.md)
- [`src_app_main.md`](src_app_main.md)
- [`src_app_result_emit.md`](src_app_result_emit.md)

### Public API headers (`include/steppe/`)

- [`include_steppe_config.md`](include_steppe_config.md)
- [`include_steppe_dates.md`](include_steppe_dates.md)
- [`include_steppe_dstat.md`](include_steppe_dstat.md)
- [`include_steppe_error.md`](include_steppe_error.md)
- [`include_steppe_extract.md`](include_steppe_extract.md)
- [`include_steppe_f3.md`](include_steppe_f3.md)
- [`include_steppe_f4.md`](include_steppe_f4.md)
- [`include_steppe_f4ratio.md`](include_steppe_f4ratio.md)
- [`include_steppe_fstat_sweep.md`](include_steppe_fstat_sweep.md)
- [`include_steppe_fstats.md`](include_steppe_fstats.md)
- [`include_steppe_qpadm.md`](include_steppe_qpadm.md)
- [`include_steppe_qpfstats.md`](include_steppe_qpfstats.md)
- [`include_steppe_qpgraph.md`](include_steppe_qpgraph.md)
- [`include_steppe_qpgraph_search.md`](include_steppe_qpgraph_search.md)

### Bindings

- [`bindings_module.md`](bindings_module.md)

---

## How to use these reviews

1. Start with the three synthesized/cross-cutting reports for the big picture.
2. Drill into individual files for line-specific flags, code snippets, and grades.
3. The most common repeated issues across the codebase are:
   - Integer-width inconsistency (`long`/`size_t`/`int`/`long long` soup).
   - Error-handling contract schizophrenia (`Status` vs exceptions vs swallowed `runtime_error`).
   - Duplicated output/emit logic bypassing the shared `result_emit` layer.
   - Over-commenting with stale scaffolding and internal ticket references.
   - Copy-paste drift in assemblers, readers, and command boilerplate.

Total reviews in this directory: **100** (97 per-file + 3 synthesized/cross-cutting).
