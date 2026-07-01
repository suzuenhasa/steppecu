# The §4 COMBINE GATE (multi-GPU f2 combine path selection)

> Source: `src/core/fstats/f2_blocks_multigpu.cpp` — the gate decided just before the
> per-device fan-out, guarding `const bool use_p2p = select_p2p_combine(resources, G);`.
> The four-term predicate is single-homed in `select_p2p_combine` (the file-local helper).
> Cross-refs: architecture.md §8 (single-source), §11.4 (capability-tiered combine), §12
> (parity law). See also `docs/design/p2p-combine.md`.

## THE §4 COMBINE GATE — decided BEFORE the fan-out

The gate depends ONLY on config + caps + G, all known up front (a CRITICAL ORCHESTRATION
POINT).

The four-term gate predicate is single-homed in `select_p2p_combine` (the file-local
helper) so it exists ONCE in code — architecture.md §8 single-source; cleanup X6/X8/B4 +
group-5 5.3. Every other site that names this gate (resources.hpp `CombinePath` /
`last_combine_path` doc, p2p_combine.hpp "the gate is the caller's" doc, config.hpp
`enable_peer_access` / `prefer_p2p_combine` docs, src/core/CMakeLists.txt module comment)
CROSS-REFERENCES it rather than restating the predicate. The §4 gate selects the OPT-IN
device-resident `cudaMemcpyPeer` combine when ALL FOUR terms hold:

```
    prefer_p2p_combine               // WHICH-PATH intent  (config.hpp)
 && enable_peer_access               // MAY-WE permission   (config.hpp)
 && gpus[0].caps.can_access_peer     // DISCOVERED probe (build_resources)
 && G >= 2                           // STRUCTURAL (dead-true here; see below)
```

(all four true on rtxbox: PRO 6000 stock-driver P2P, `canAccessPeer==1` both ways), else
it DEGRADES to the host-staged fixed-order combine baseline. Both tiers place the SAME
bytes at the SAME disjoint offsets in the SAME fixed `g=0..G-1` order, and are therefore
BIT-IDENTICAL to each other and to the single-GPU reference (the gate is parity-NEUTRAL —
the transport only moves bytes; software fixes the order; architecture.md §11.4, §12).
NEVER an NCCL AllReduce. The chosen path is recorded OUT-OF-BAND on `Resources` (NEVER on
the numeric `F2BlockTensor`; cleanup §(2).2), and a genuine degrade (P2P requested +
permitted but peer access unavailable) emits the architecture-mandated tagged WARN via
the non-throwing path (derived from `requested_p2p_combine && !use_p2p`, so it cannot
drift from the gate).

## WHY THE GATE MOVED BEFORE THE FAN-OUT (M4.5 cure, doc §4 Item 1)

The P2P arm now needs a DIFFERENT fan-out — the DEVICE-RESIDENT compute that leaves each
partial on its device (`compute_multigpu_partials_resident` -> `DevicePartial`) feeding
the device-resident combine — while the host-staged baseline keeps the EXACT existing
host-partial fan-out (`compute_multigpu_partials` -> `combine_f2_partials_host`). The gate
depends only on config + caps + G (all known up front), so it is decided here, BEFORE
either fan-out, and forks into the matching resident-vs-host pair.

## THE FOUR TERMS (spelled in `select_p2p_combine` / `requested_p2p_combine`)

- **`prefer_p2p_combine`** — the WHICH-PATH knob: once peer access IS permitted and
  available, prefer the device-resident combine over the host-staged baseline.
- **`enable_peer_access`** — the MAY-WE knob: "whether the backend is permitted to call
  `cudaDeviceEnablePeerAccess` at all" (config.hpp). The device-resident combine
  `combine_f2_partials_resident` DOES call `cudaDeviceEnablePeerAccess`
  (cuda/p2p_combine.cu), so taking that path with `enable_peer_access==false` would
  directly VIOLATE the veto the user set. The gate ANDs it in so a user who set
  `enable_peer_access=false` (forbid the enable) is honored and the enable is reached only
  WITH permission (cleanup CT2 / config C-1/K-2; config.hpp OVERRIDE-KNOB banner). The two
  override-intent knobs are DISTINCT and BOTH must permit P2P.
- **`can_access_peer`** — the DISCOVERED capability probe (set ONCE at `build_resources`,
  never re-probed here).
- **`G >= 2`** — STRUCTURAL, and DEAD-TRUE at this point: the `G==1` single-GPU fast-path
  returned at the top of this function (the `if (G == 1)` early return), so control only
  reaches this gate with `G >= 2`. The term is spelled in the gate so the CODE MATCHES the
  predicate documented across the seam (cleanup X6: it was documented as 4-term in five
  files while shipping as 3-term — a latent hazard if the gate were ever lifted into a
  reusable `select_combine_path(resources)` that could be reached at `G==1`). It changes NO
  reached path (parity-NEUTRAL: G is always >= 2 here, so the AND is identity).

The gate is parity-NEUTRAL either way (both transports are bit-identical, §12), so the
four-term AND only changes WHICH transport runs, never a reported number.
