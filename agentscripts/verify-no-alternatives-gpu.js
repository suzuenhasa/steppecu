export const meta = {
  name: 'verify-no-alternatives-gpu',
  description: 'Adversarially re-verify every "not possible on consumer 5090" verdict (chase aikitoria / NCCL #1637 P2P), assume an RTX PRO 6000 full-host is available, and design capability-tiered: build for the capable path, graceful EXPLICIT fallback on the budget 5090.',
  phases: [
    { title: 'Investigate', detail: 'parallel: P2P/NCCL (aikitoria/#1637), GPUDirect Storage re-check, adversarial sweep of remaining negatives, RTX PRO 6000 re-scope' },
    { title: 'Synthesize', detail: 'verdicts FLIP/STAND, capability tiers (capable path vs budget-5090 fallback w/ explicit message), per-milestone design changes' },
  ],
}

const CONTEXT = `
PROJECT: steppe — CUDA-13/Blackwell (sm_120) GPU reimplementation of ADMIXTOOLS 2 f-statistics. Goal: keep the pipeline GPU-DOMINANT (no weak-host-CPU bottleneck) and maximize multi-GPU throughput WITH numerical parity (§12).

WHY THIS RE-VERIFICATION: a prior research workflow concluded several things are "not possible / not worth it on the vast consumer RTX 5090 container." The user pushes back: (1) Community solutions exist for consumer GPUs that official docs call unsupported — they cite NCCL issue https://github.com/NVIDIA/nccl/issues/1637 and the dev https://github.com/aikitoria (the consumer-GPU P2P-enablement scene: geohot/tinygrad RTX 4090 P2P patch via the open GPU kernel modules, and Blackwell/5090 successors). (2) We have an RTX PRO 6000 Blackwell we can set up with full host, no problem — so "needs a professional card / host kernel control" is NOT a blocker.

YOUR STANCE: ADVERSARIAL, skeptical-of-"officially-unsupported". For every negative verdict, ASSUME a workaround exists and hunt for it (community kernel patches, alternative CUDA APIs, unofficial driver flags, the open-gpu-kernel-modules tree, issue/forum threads). Conclude "genuinely impossible" only after real effort, stating exactly what you checked. When a workaround exists, be CONCRETE: repo/commit, kernel/driver version, exact steps, stability/risk.

DESIGN PHILOSOPHY (user directive, fold into recommendations): treat the vast consumer 5090 as the BUDGET / FALLBACK box, and BUILD PRIMARILY FOR THE CAPABLE PATH — full-host RTX PRO 6000 + enabled P2P + GDS + NVLink-if-present ("the cool shit"). Every capability must be RUNTIME-DETECTED and degrade GRACEFULLY on the budget box with an EXPLICIT logged/tagged reason, e.g. "P2P device-combine unavailable (no P2P/NVLink on this GPU) -> fell back to host-staged fixed-order combine". The user wants OPTIONS: capable path when hardware allows, clearly-tagged fallback otherwise. Parity (§12) must hold on BOTH paths.

HARDWARE:
- Budget/daily box: vast.ai RENTED consumer 2x RTX 5090 (Blackwell GB202, sm_120, CUDA 13.0, driver 580, Docker over OverlayFS, NO host kernel control).
- Capable box (assume available, set up, no problem): RTX PRO 6000 Blackwell (GB202-class, sm_120, 96 GB, PROFESSIONAL, FULL HOST — control kernel/driver/filesystem). Both Blackwell sm_120.

NEGATIVE VERDICTS TO RE-VERIFY (re-check EACH, on BOTH boxes):
 1. True GPUDirect Storage / cuFile — prior: dead on vast 5090 (GeForce policy + no host kernel module + OverlayFS); falls to POSIX-pread compat mode.
 2. P2P / NVLink between the two consumer 5090s — NVIDIA disables P2P-over-PCIe on GeForce; NCCL degrades/hangs (the #1637 / aikitoria topic).
 3. Blackwell hardware Decompression Engine — prior: datacenter-only (B200/B300), absent on GB202.
 4. Nsight Compute (ncu) perf counters — prior: restricted on consumer + container.
 5. O_DIRECT on overlay FS, RLIMIT_MEMLOCK pinned-memory cap — prior: container limitations.
 6. cuSOLVERMp distributed solve — prior: deferred (qpAdm matrices tiny; overhead).

GROUND IN OUR DESIGN — read: /home/suzunik/steppe/docs/architecture.md §11.1 (streaming), §11.4 (single-node multi-GPU: the PARITY combine is HOST-SIDE fixed-order, NOT NCCL AllReduce, because AllReduce order varies with GPU count), §12 (determinism/parity); /home/suzunik/steppe/docs/TODO.md (the "Keeping it GPU-dominant" section + the "Decided against" list you are re-verifying).

USE WEB HEAVILY: load WebFetch/WebSearch via ToolSearch. FETCH the two cited URLs first. Then research the consumer-GPU P2P lineage, NCCL-on-consumer-Blackwell, NVMe-P2PDMA GDS, open-gpu-kernel-modules P2P/large-BAR patches, RTX PRO 6000 Blackwell capabilities (P2P/NVLink? GDS? decompression engine?). Cite sources (repo+commit, doc, thread).`

const FIND_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['topic','prior_verdict','what_was_checked','alternative_found','feasibility','requirements','what_it_buys_steppe','risk','revised_verdict','sources'],
  properties: {
    topic: { type: 'string' },
    prior_verdict: { type: 'string' },
    what_was_checked: { type: 'string', description: 'specific sources/approaches investigated — repos, kernel versions, APIs, threads' },
    alternative_found: { type: 'string', description: 'the concrete workaround/path (repo/commit, exact steps) or "none found after checking X"' },
    feasibility: { type: 'string', enum: ['vast-5090','rtx-pro-6000','both','neither'] },
    requirements: { type: 'string', description: 'host control / kernel patch / driver flag / FS needed' },
    what_it_buys_steppe: { type: 'string', description: 'concrete benefit (M4.5 multi-GPU / M5 ingest / profiling), incl. whether it preserves the §12 parity contract' },
    risk: { type: 'string', description: 'stability, maintenance, breakage-on-update, support' },
    revised_verdict: { type: 'string', enum: ['FLIPS','STANDS','CONDITIONAL'] },
    sources: { type: 'array', items: { type: 'string' } },
  },
}
const INVEST_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension','findings','headline'],
  properties: {
    dimension: { type: 'string' },
    findings: { type: 'array', items: FIND_SCHEMA },
    headline: { type: 'string', description: 'one-sentence bottom line for this dimension' },
  },
}

const DIMS = [
  { key: 'p2p-nccl', title: 'Consumer-Blackwell P2P + NCCL (the aikitoria / NCCL #1637 lead)',
    focus: `FETCH https://github.com/NVIDIA/nccl/issues/1637 and https://github.com/aikitoria first. Determine exactly what #1637 reports (NCCL on 2x RTX 5090 — P2P disabled on GeForce so NCCL fails/hangs/falls back to slow SHM?) and what aikitoria contributes (a patch enabling P2P on consumer Blackwell via the open GPU kernel modules — trace the lineage: geohot/tinygrad RTX 4090 P2P patch -> open-gpu-kernel-modules large-BAR/P2P enablement -> 5090/Blackwell port). On 2x RTX 5090 with FULL HOST (we can build+load patched open kernel modules and set driver flags): can we enable working P2P + NCCL? Quantify the win for steppe M4.5 multi-GPU. CRITICAL STEPPE ANGLE: our parity design deliberately does the f2_blocks combine HOST-SIDE in fixed device order (NOT NCCL AllReduce, which breaks parity across GPU count, §12). Does P2P let us instead do a PARITY-EXACT device-to-device combine — one GPU pulls the peer device partial f2_blocks via P2P and sums in fixed order ON-DEVICE — keeping the combine GPU-dominant AND bit-identical? Also: does the RTX PRO 6000 Blackwell support P2P/NVLink OFFICIALLY (clean, no patch)? And what is the explicit budget-5090 FALLBACK if P2P is unavailable (the existing host-staged fixed-order combine) and the message it should log?` },
  { key: 'gds-recheck', title: 'GPUDirect Storage — re-verified across all paths + both boxes',
    focus: `Re-verify the GDS verdict. Three sub-paths: (a) NVMe-P2PDMA mode added in CUDA 12.8 (direct PCIe peer NVMe<->GPU DMA WITHOUT nvidia-fs — needs a recent host kernel + O_DIRECT-capable FS + x86_64): does it work on a FULL-HOST 5090 box (we control the kernel + put data on ext4/XFS, not overlay)? (b) any community/unofficial GeForce GDS enablement? (c) the OFFICIAL path on the RTX PRO 6000 (professional CC>6 + full host + GDS-qualified FS = clean supported GDS). For steppe: which path delivers real DMA-bypass for the large contiguous .f64 / M7 f2_blocks cache (the right GDS-shaped target), and on which box. KEY: with FULL HOST on the 5090 box, two of the three prior disqualifiers (host kernel module, overlay FS) evaporate — does only the GeForce policy lock remain, and is THAT bypassable (patched kernel module / the NVMe-P2PDMA path which sidesteps nvidia-fs entirely)? State the explicit budget-box fallback (pinned double-buffer) and its logged reason.` },
  { key: 'adversarial-sweep', title: 'Adversarial sweep of the remaining negative verdicts',
    focus: `Re-verify each remaining "not possible" claim, hunting the workaround: (1) Blackwell hardware Decompression Engine — truly absent on GB202 (both 5090 and RTX PRO 6000), or any DMA/HW path? if genuinely silicon-absent, say so. (2) Nsight Compute perf counters — does NVreg_RestrictProfilingToAdminUsers=0 (full host) fully unlock ncu on the 5090 / RTX PRO 6000? (3) O_DIRECT + RLIMIT_MEMLOCK — both moot on a full-host real-FS box? confirm. (4) cuSOLVERMp — worth reconsidering IF P2P/NVLink is enabled, or still pure overhead for qpAdm tiny matrices regardless? For each: concrete workaround vs genuinely-impossible, on which box, and the explicit budget-5090 fallback + logged reason. Be the skeptic who refuses "unsupported" without checking the open-source driver + community.` },
  { key: 'rtx-pro-6000-rescope', title: 'RTX PRO 6000 Blackwell as the primary capable box (assume available, full host)',
    focus: `Assume the RTX PRO 6000 Blackwell (sm_120, 96 GB, professional, full host) is the PRIMARY box we build for; the consumer 5090 is the budget fallback. Concretely: what does the RTX PRO 6000 unlock the 5090 cannot? Verify: (a) official GPUDirect Storage; (b) full ncu/nsys; (c) does it support P2P/NVLink officially (clean multi-GPU WITHOUT the consumer patch)? does the RTX PRO 6000 Blackwell even HAVE NVLink or is it PCIe-only? (d) 96 GB VRAM — which steppe workloads that exceed a single 5090 32 GB now FIT single-GPU? compute which population counts P fit f2_blocks in 96 GB (f2_blocks bytes = P*P*n_block*8; n_block ~757; e.g. P=2416 -> ~33.5 GB, P=4266 -> ?) and whether that defers the need for M5 streaming / M4.5 sharding for large P. (e) sm_120 so Ozaki speedups ARE representative (unlike Ada). RECOMMEND the role split (RTX PRO 6000 = primary perf+multi-GPU+GDS; vast = cheap parallel dev + the budget-fallback test target) and what in the M4.5/M5/M7/Phase2 plan changes if we design for "RTX PRO 6000 primary, 5090 fallback".` },
]

phase('Investigate')
const investigations = (await parallel(DIMS.map(d => () =>
  agent(
    `You are a skeptical GPU systems researcher re-verifying a negative verdict. ${CONTEXT}\n\nYOUR DIMENSION: ${d.title}\n${d.focus}\n\nFetch the cited URLs + do real web research, ground in our architecture, return findings per the schema. revised_verdict must be FLIPS / STANDS / CONDITIONAL with concrete evidence. Refuse to accept "officially unsupported" as "impossible" without checking the community + open-source driver. For every lever, name the explicit budget-5090 fallback and the reason it would log.`,
    { schema: INVEST_SCHEMA, label: `verify:${d.key}`, phase: 'Investigate' }
  )
))).filter(Boolean)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['aikitoria_nccl_finding','verdicts_that_flip','verdicts_that_stand','conditional','capability_tiers','rtx_pro_6000_rescope','design_changes','recommendation','summary'],
  properties: {
    aikitoria_nccl_finding: { type: 'string', description: 'concrete: what #1637/aikitoria is, whether consumer-5090 P2P+NCCL is enableable with full host, and what it buys M4.5 (incl. the parity-exact device-to-device combine question)' },
    verdicts_that_flip: { type: 'array', items: { type: 'string' } },
    verdicts_that_stand: { type: 'array', items: { type: 'string' } },
    conditional: { type: 'array', items: { type: 'string' } },
    capability_tiers: { type: 'array', description: 'per major lever: the capable-path design + the explicit budget-5090 fallback + the message it logs', items: {
      type: 'object', additionalProperties: false, required: ['lever','capable_path','budget_5090_fallback','fallback_message','parity_preserved'],
      properties: { lever: { type: 'string' }, capable_path: { type: 'string' }, budget_5090_fallback: { type: 'string' }, fallback_message: { type: 'string' }, parity_preserved: { type: 'string' } } } },
    rtx_pro_6000_rescope: { type: 'string', description: 'what it unlocks + recommended box role-split + which large-P workloads now fit in 96 GB' },
    design_changes: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['milestone','change'], properties: { milestone: { type: 'string' }, change: { type: 'string' } } } },
    recommendation: { type: 'string', description: 'decisive: build-for-capable-path design with runtime capability detection + explicit tagged fallback; what stays portable vs opt-in fast-path' },
    summary: { type: 'string' },
  },
}
const synthesis = await agent(
  `You are the synthesis lead. Below are four adversarial re-verifications of the "not possible on consumer 5090" verdicts (chasing aikitoria/NCCL-#1637 P2P) re-scoped around an available RTX PRO 6000 Blackwell full-host as the PRIMARY box. ${CONTEXT}\n\nINVESTIGATIONS (JSON):\n${JSON.stringify(investigations, null, 1)}\n\nProduce a decisive synthesis per the schema. CENTER it on the user directive: BUILD FOR THE CAPABLE PATH (RTX PRO 6000 + P2P + GDS), with the consumer 5090 as a clearly-tagged graceful FALLBACK. The capability_tiers array is the key deliverable — for each major lever (multi-GPU f2_blocks combine, disk->GPU ingest, profiling, cache decompression) give the capable_path, the budget_5090_fallback, the exact fallback_message it logs (e.g. "P2P combine unavailable: no peer access on this GPU -> host-staged fixed-order combine"), and how parity holds on BOTH. Give the concrete aikitoria/NCCL answer (incl. parity-exact device-to-device combine), the verdict flips/stands/conditional, the RTX PRO 6000 re-scope, per-milestone design changes, and a clear recommendation. Do not hand-wave a workaround the findings did not substantiate.`,
  { schema: SYNTH_SCHEMA, label: 'synthesize', phase: 'Synthesize' }
)

return { investigations, synthesis }