# Known issue — multi-GPU rotation host-bounce cap (DEFERRED)

> Source: `src/core/qpadm/model_search.cpp` — the `F2Replication` broadcast
> (`replicate_f2_to_all_devices`). The single-GPU path (`run_qpadm_search` G==1 fast path) is
> the SUPPORTED / recommended path. The 1-line DEFERRED guard is retained at the call site;
> the perf narrative lives here. Memory: [multi-GPU is PARKED] / [m4.5 multigpu slowdown].

## TODO(multigpu-host-bounce)

KNOWN PROBLEM — multi-GPU rotation is bounce-capped on no-P2P cards. DEFERRED. The
single-GPU path (`run_qpadm_search` G==1 fast path) is the SUPPORTED / recommended path for
now; this `G>=2` replication is CORRECT + deterministic (G1==G2 bit-identical) but
throughput-suboptimal — do NOT rely on it for speed until the bounce is removed.

- The doc once said "kB-MB scale" — WRONG. MEASURED on REAL AADR (P=600, n_block=757):
  this replication is ~8.72 GB through host (2x D2H + 2x H2D), ~3.8 s COLD. At 9086 real
  models G2/G1 only reached 1.21x and never hit 1.5x in the swept range — the fixed
  host-bounce dominates until impractically large N.
- **ROOT CAUSE:** consumer RTX 5090s have GPU<->GPU P2P disabled (`caps.can_access_peer ==
  false`), so the device-resident `cudaMemcpyPeer` fast-path is unavailable and we fall back
  to the host-staged transport (`upload_f2_blocks_to_device` = `to_host()`+H2D).
  P2P-capable cards (e.g. RTX PRO 6000) avoid it.
- **DEEPER WART (the "weird orchestration"):** the PRECOMPUTE shards SNP tiles across GPUs
  then COMBINES the partials down to ONE device; the rotation then re-broadcasts that single
  f2 back OUT to the other device here — a gather-then-scatter round-trip.
- **FIX TO ELIMINATE (not reduce):** give the rotation a PER-DEVICE PRECOMPUTE path — each
  GPU builds its OWN full f2 directly from the genotype stream (zero cross-GPU transfer,
  ~2.6 s parallel, CHEAPER than the 3.8 s bounce, works on any hardware incl. no-P2P). Then
  no `to_host()`, no upload, no combine, no replicate. (Avoid the band-aid of merely pinning
  the staging: warm pinned DMA drops a *repeat* replica to ~320 ms, but a single rotation
  pays the COLD ~3.8 s once.)
