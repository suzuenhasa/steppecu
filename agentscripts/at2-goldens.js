export const meta = {
  name: 'at2-goldens',
  description: 'M(fit-0): stand up the ADMIXTOOLS 2 GOLDENS — the Phase-2 acceptance oracle (architecture.md §12/§13). On box5090: install R + admixtools, run extract_f2 on the real AADR data for a small pop set, run qpadm() on ONE well-determined 2-way model with boot=FALSE (deterministic), and PIN the golden (est weights / se / z / p + the intermediate f4 matrix X and the Q covariance for the OQ-3 block-weight cross-check), recording the FULL §12 reproducibility metadata (R version, admixtools version, blgsize, allsnps/maxmiss, seed, the dataset + target/left/right). Pull the golden into the repo (tests/reference/goldens/at2/) + a manifest doc, committed. This is the reference M(fit-1) (the first GLS fit) will be bit-/tolerance-checked against. Generate-on-box -> INDEPENDENT verdict (goldens real + metadata complete + model well-determined) commits.',
  phases: [
    { title: 'Generate', detail: 'box5090: install R+admixtools, extract_f2 + qpadm() one 2-way model (boot=FALSE), capture golden + X/Q + §12 metadata' },
    { title: 'Record', detail: 'independent verdict: pull goldens into the repo + manifest, verify real+complete, commit' },
  ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'

const STD = [
  'PROJECT: steppe = GPU reimplementation of ADMIXTOOLS 2 f-statistics. Phase-1 precompute (f2_blocks) is DONE; Phase 2 (the qpAdm fit engine, S3-S8) is being designed/built (docs/design/fit-engine.md). The acceptance gate for the fit engine is BIT/TOLERANCE PARITY against ADMIXTOOLS 2 goldens (architecture.md §12 PARITY LAW + §13). This milestone M(fit-0) creates the FIRST such golden — the reference M(fit-1) (the first qpAdm GLS fit, CpuBackend, native FP64, ONE model) will be validated against.',
  'BOX = box5090 (vast 2x RTX 5090, UP; CPU is all AT2 needs — admixtools is R, no GPU). ' + SSH + ' (alias); flaky network -> run long installs/extract_f2 DETACHED on the box + poll a /tmp logfile. The REAL AADR data is at /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.{geno,ind,snp} (EIGENSTRAT/PACKEDANCESTRYMAP; ~4266 pops in the .ind). NO synthetic data.',
  'WHAT TO PRODUCE (the golden): (1) install R + the admixtools package (apt-get r-base + R: install.packages or remotes::install_github("uqrmaie1/admixtools"); resolve deps; verify library(admixtools) loads + record packageVersion). (2) extract_f2(prefix, outdir, pops=<the small pop set of the model + a few outgroups>, blgsize=0.05, maxmiss=0, ...) on the raw AADR. (3) qpadm(f2_dir, target, left=<2 sources>, right=<>=4 outgroups>, boot=FALSE) for ONE WELL-DETERMINED 2-way model (pick a textbook-feasible model whose pops EXIST in the .ind — e.g. a documented European/steppe 2-way, or an AT2-tutorial example present in AADR; CONFIRM the pops are in the .ind first via grep). (4) CAPTURE: the qpadm output (weights/est, se, z, the p-value/tail prob, the rank-test stats), AND the intermediate f4 matrix X and the jackknife covariance Q (qpadm() returns/exposes these or recompute via f4blockdat_to_f4/f2blocks_to_f4blocks) — these are the OQ-3 (block-weight) + OQ-1 (ALS) cross-check goldens the design needs. (5) RECORD the full §12 metadata: R version, admixtools packageVersion, blgsize, maxmiss/allsnps, any seed/RNGkind, the dataset prefix + sha, and the exact target/left/right pop lists.',
  'HONESTY: this is the ORACLE — it must be REAL (actually run on the box, real AADR, real admixtools). Do NOT fabricate any golden number. If the model is not well-determined (e.g. infeasible p, or rank issues), pick another and SAY which + why. Cite the AT2 qpadm()/extract_f2 API (web search ok via ToolSearch select:WebSearch,WebFetch for the recipe + a known-feasible model).',
].join('\n')

phase('Generate')
const gen = await agent([
  'You are a pop-gen-aware engineer standing up the ADMIXTOOLS 2 golden oracle on box5090. Do the box work (install R+admixtools, extract_f2, qpadm), capturing a REAL golden + metadata. Long steps DETACHED + poll. Use web search for the AT2 recipe + a known-feasible 2-way model.', STD, '',
  'STEPS: (1) ' + SSH + " 'R --version 2>/dev/null | head -1 || echo NO_R; ls /workspace/data/aadr/raw/' — check R + the data. (2) Install R + admixtools if needed (apt-get update && apt-get install -y r-base; then Rscript -e 'install.packages(\"remotes\"); remotes::install_github(\"uqrmaie1/admixtools\", upgrade=\"never\")' — DETACHED to /tmp/at2install.log, poll; admixtools pulls many deps + compiles, can take 10-30min). Verify: Rscript -e 'library(admixtools); packageVersion(\"admixtools\"); R.version.string'. (3) Pick a well-determined 2-way model whose pops are in the .ind (grep the .ind col-3 for candidates; confirm presence). (4) Write an Rscript that: extract_f2(prefix, outdir, pops=c(target,sources,rights), blgsize=0.05, maxmiss=0); res = qpadm(outdir, target, left, right, boot=FALSE); print res$weights (est/se/z), res$popdrop / res$rankdrop (the p-value + rank stats); ALSO dump the f4 matrix X and the covariance Q if reachable. Run it DETACHED + poll. (5) Capture ALL outputs + the §12 metadata verbatim.",
  'Return: the EXACT golden numbers (weights est/se/z, p/tail, rank stats; X and Q if captured) as captured from the box, the model (target/left/right), the §12 metadata (R ver, admixtools ver, blgsize, maxmiss, seed, dataset), and the paths of any golden files written on the box (so the verdict can pull them). If install or the fit failed, report exactly what + how far you got — do NOT fabricate a golden.',
].join('\n'), { label: 'generate:at2-golden', phase: 'Generate' })

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','admixtools_installed','golden_is_real','model','at2_version','golden_committed','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: admixtools really installed on box5090; a REAL qpadm() golden was produced on the real AADR for a well-determined 2-way model (boot=FALSE); the golden values + the §12 metadata are captured; and they are written into the repo (tests/reference/goldens/at2/) + committed' },
    admixtools_installed: { type: 'boolean', description: 'library(admixtools) loads on box5090 (you verified the version)' },
    golden_is_real: { type: 'boolean', description: 'the qpadm est/se/z/p come from an actual run on real AADR (not fabricated) — you re-checked the run log / re-ran a spot value' },
    model: { type: 'string', description: 'the target / left (sources) / right (outgroups) of the golden model + whether it is well-determined (feasible p, full rank)' },
    at2_version: { type: 'string', description: 'admixtools packageVersion + R version (the §12 metadata)' },
    golden_committed: { type: 'boolean', description: 'the golden values + manifest (with full §12 metadata) are in the repo and committed' },
    commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the golden weights/se/z/p + what intermediate (X/Q) was captured; for FAIL exactly what blocked it (install? infeasible model? capture?)' },
  },
}
phase('Record')
const verdict = await agent([
  'You are the INDEPENDENT VERDICT/recorder for the AT2 golden. Verify the golden is REAL (not fabricated) and complete, then WRITE it into the repo + commit. The generate agent reported:\n<<<\n' + (gen || '(generate agent died)') + '\n>>>\n\n' + STD + '\n\n' +
  'DO: (1) verify on the box yourself: ' + SSH + " 'Rscript -e \"library(admixtools); packageVersion(\\\"admixtools\\\")\" 2>&1 | tail -2' (admixtools really installed) and re-read the run log / re-print one golden value to confirm it is real (NOT fabricated). (2) Pull the golden into the repo: create tests/reference/goldens/at2/ with the golden VALUES (qpadm weights est/se/z, p/tail, rank stats; X and Q if captured) as a committed data file (JSON or a documented text format the future M(fit-1) parity test can read) + a manifest README recording the FULL §12 metadata (R ver, admixtools ver, blgsize, maxmiss/allsnps, seed/RNGkind, dataset prefix + a sha if cheap, target/left/right). (3) PASS only if: admixtools really installed; the golden is real + well-determined; values + §12 metadata captured + written to the repo. \n\nON PASS: cd " + R + " && git add ONLY tests/reference/goldens/at2/ (+ any small manifest doc) — NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md; commit with a ROADMAP §6 message (the model + the AT2/R versions + that it is the M(fit-1) acceptance oracle) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash.\nON FAIL: do NOT commit a fake golden; report exactly what blocked it.\nReturn the structured verdict.",
].join('\n'), { schema: VERDICT_SCHEMA, label: 'record:at2-golden', phase: 'Record' })

if (verdict && verdict.pass) log('+++ AT2 GOLDEN PINNED ' + verdict.commit_hash + ' — model ' + verdict.model + ' (admixtools ' + verdict.at2_version + ')')
else log('--- AT2 GOLDEN FAILED (' + (verdict ? verdict.note : 'agent died') + ') — HALT; human takes over')
return { gen, verdict }
