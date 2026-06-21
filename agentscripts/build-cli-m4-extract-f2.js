export const meta = {
  name: 'build-cli-m4-extract-f2',
  description: 'STEP 2 / M(cli-4): build `steppe extract-f2` — the genotype->f2-dir command that unlocks running ANY study (Haak 2015 etc.) from the raw AADR, the last hard blocker for end-to-end CLI use. Wires the EXISTING engine: io readers (EIGENSTRAT/PACKEDANCESTRYMAP) + --pops selection (names | auto-top-K | min-N) + filter flags -> decode_af -> assign_blocks -> compute_f2_blocks_multigpu_device/_tiered -> WRITE an f2 dir (the NEW STPF2BK1 writer — M(cli-1) only built the reader — with REAL vpair so the F1 NA-handling works, + pops.txt name<->index sidecar + meta.json provenance). Then `steppe qpadm --f2-dir <that dir>` runs (M(cli-1), built). Phases: LOCATE+design (find the raw AADR on the box; HALT if absent) -> fixer -> build-repair -> adversarial verdict. GOLDEN GATE (real AADR, NO synthetic): extract-f2 on the raw AADR for a golden pop set (e.g. golden_fit0 maxmiss=0, and golden_fitNA maxmiss=0.99 for the NA path) -> qpadm -> reproduces the golden end-to-end; and the extracted f2 matches the steppe precompute. GPU-only; app a plain CXX target (CUDA PRIVATE, arch-grep holds); follow cli-bindings.md + NAMING-STYLE-STANDARD; CUDA-doc verify; commit on green; HALT-on-fail.',
  phases: [ { title: 'Locate+design' }, { title: 'Build extract-f2' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -30 && echo === CTEST === && ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const ARCHGREP = SSH + " 'cd /workspace/steppe && ! grep -rnE \"cuda_runtime|cublas_v2|cusolver|<cuda\" src/app 2>/dev/null && echo APP-CUDA-FREE-OK || echo APP-CUDA-LEAK'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13/Blackwell (sm_120) C++20 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main. The fit BACKEND is FINISHED + golden-gated; the CLI exists through M(cli-1) (`steppe qpadm` on an f2 dir, golden-gated). THIS is M(cli-4): `steppe extract-f2` (genotypes -> f2 dir) — the last hard blocker to run real studies end-to-end. THE CONTRACT is docs/design/cli-bindings.md §4 (extract-f2) + §4.3 (the f2-dir format) — READ IT. Standards: docs/architecture.md (§4 layering [app plain CXX, CUDA PRIVATE to steppe_device, io leaf], §5 the precompute S0-S2, §9 ConfigBuilder, §10 Status/no-printf-in-lib), docs/cleanup/bigrefactor/NAMING-STYLE-STANDARD.md.',
  'THE ENGINE ALREADY EXISTS — extract-f2 is mostly WIRING + the new writer: (1) io readers (src/io/{eigenstrat_format,geno_reader,snp_reader,ind_reader,genotype_tile}.hpp) decode genotypes + the .ind pop labels; (2) the filters (src/io/filter/*); (3) src/core/internal/decode_af.hpp -> per-pop Q/V/N; (4) src/core/domain/block_partition_rule.hpp assign_blocks (blgsize); (5) src/core/fstats/f2_blocks_multigpu.hpp compute_f2_blocks_multigpu_device / _tiered -> DeviceF2Blocks / F2BlocksOut (the precompute; CUDA-free seam, golden-gated through M5). The NEW piece: the STPF2BK1 f2-dir WRITER (M(cli-1) f2_dir_io.cpp built only the READER) — write <out>/f2.bin (STPF2BK1 per src/device/f2_disk_format.hpp, with the REAL vpair region — NOT zeros; the F1 missing-block/NA handling reads vpair==0), <out>/pops.txt (the P pop labels in P-axis index order), <out>/meta.json (provenance: steppe version+SHA, precision_tag, blgsize, maxmiss, the dataset shas, pop selection). Mirror the reader f2_dir_io.cpp + the test write_f2_dir (tests/cli/test_cli_qpadm.cpp) but write REAL vpair from the precompute (F2BlockTensor.vpair / DeviceF2Blocks.to_host()).',
  'COMMAND (cli-bindings.md §4): `steppe extract-f2 (--geno G --snp S --ind I | --prefix P) --pops <names,..|auto-top:K|min-n:N> --out DIR [--blgsize 0.05 --maxmiss 0 --maf .. --auto-only --precision emu40|fp64 --device 0 --dry-run]`. --pops resolves names against the .ind labels (unknown name => Status::InvalidConfig fail-fast). --out writes the f2 dir the existing `steppe qpadm --f2-dir` consumes. GPU-only (memory cpu-is-test-only): the precompute runs on the GPU via build_resources(DeviceConfig); NO --device cpu. Single-GPU default.',
  'REAL DATA ONLY (memory real-data-only-all-results) + the existing golden test pattern (NO synthetic, NO smoke): the golden gate runs extract-f2 on the REAL raw AADR on the box and asserts the result reproduces a committed AT2 golden. The raw AADR (geno/snp/ind, dataset v66.p1_HO.aadr.patch.PUB, the shas in tests/reference/goldens/at2/golden_*.json metadata) lives on the box (the precompute M1-M5 + the golden generators used it) — LOCATE it first (check /workspace for *.geno/*.snp/*.ind, the golden generator scripts tests/reference/goldens/at2/scripts/*, any build_run.sh). If the raw AADR is NOT on the box, STOP and report (do not synthesize data, do not fake the gate).',
  'GATE: Release build clean (-Werror, -DSTEPPE_BUILD_CLI=ON); the existing ctest STILL green (no regression); the app-is-CUDA-free arch-grep holds; AND the new end-to-end test: `steppe extract-f2` on the raw AADR for the golden_fit0 pop set (target England_BellBeaker + its left/right, blgsize=0.05, maxmiss=0) writes an f2 dir, then `steppe qpadm --f2-dir <dir>` reproduces golden_fit0 (weights/p/chisq within the existing tiers); + the extracted f2 matches the steppe precompute (and ideally the committed f2_fit0 fixture) within tol. If feasible also exercise golden_fitNA (maxmiss=0.99) to prove the real-vpair NA path. RELEASE/NDEBUG/-Werror; core dumps cleared per build.',
  'BOX = box5090 (2x RTX 5090, sm_120, CUDA 13). ' + SSH + ' (alias); nvcc -> ' + PATHENV + ' . build-rel, RELEASE only. NOTHING builds locally. Commit on green (the verdict commits).',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD at start (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+ctest (' + BUILD + '); arch-grep (' + ARCHGREP + '). Do NOT commit (the verdict commits). NO synthetic data. Wire the new test into tests/CMakeLists.txt.'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry once'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

// 1) LOCATE the raw AADR + design (read-only). HALT the whole build if the AADR is absent.
phase('Locate+design')
const LOC_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['aadr_found','aadr_paths','approach','note'],
  properties: {
    aadr_found: { type: 'boolean', description: 'the raw AADR geno+snp+ind (v66.p1_HO) is present + readable on the box' },
    aadr_paths: { type: 'string', description: 'the geno/snp/ind paths on the box (or empty if not found)' },
    approach: { type: 'string', description: 'the concrete build approach: the io->decode->precompute->writer wiring, the STPF2BK1 writer w/ real vpair, the --pops resolution, the golden-gate plan' },
    note: { type: 'string', description: 'risks / what the golden gate will run / anything blocking' },
  },
}
const loc = await tryAgent([
  'You are scoping M(cli-4) `steppe extract-f2` for steppe AND locating the raw AADR on the box. READ-ONLY (no edits, no build). Read docs/design/cli-bindings.md §4/§4.3, the io readers (src/io/*), src/core/internal/decode_af.hpp, src/core/domain/block_partition_rule.hpp, src/core/fstats/f2_blocks_multigpu.hpp (the precompute entry signatures), src/app/f2_dir_io.cpp (the reader to mirror for the writer), src/device/f2_disk_format.hpp (STPF2BK1), the existing test write_f2_dir (tests/cli/test_cli_qpadm.cpp), and the golden generator scripts tests/reference/goldens/at2/scripts/* (for the exact AADR path + params). THEN on the box (' + SSH + '): LOCATE the raw AADR — `find /workspace -maxdepth 5 \\( -name "*.geno" -o -name "*.snp" -o -name "*.ind" \\) 2>/dev/null | grep -vi build | head -40`, and check the golden generators / any build_run.sh for the dataset path + confirm the sha matches the golden metadata.', STD, '',
  'Return the structured result: aadr_found + aadr_paths (the real geno/snp/ind on the box), the concrete build approach (wiring + the STPF2BK1 writer with REAL vpair + --pops resolution + the golden-gate plan), and risks. If the AADR is NOT found, set aadr_found=false and explain — the build will HALT.',
].join('\n'), { schema: LOC_SCHEMA, label: 'locate:aadr', phase: 'Locate+design' })

if (!loc || !loc.aadr_found) {
  log('HALT: raw AADR not located on the box — extract-f2 cannot be golden-gated without real data. ' + (loc ? loc.note : 'locator died'))
  return { halted: true, reason: 'raw AADR not found on box', loc }
}
log('AADR located: ' + loc.aadr_paths + ' — building extract-f2')

// 2) BUILD extract-f2
phase('Build extract-f2')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','diff_real','gpu_only','layering_ok','writer_real_vpair','no_synthetic','goldens_green','extract_reproduces_golden','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: real extract-f2 per cli-bindings.md + GPU-only + app CUDA-free (arch-grep) + the STPF2BK1 writer emits REAL vpair (not zeros) + NO synthetic + existing ctest green + the new end-to-end test (extract-f2 on real AADR -> qpadm reproduces golden_fit0) passes + Release build clean -DSTEPPE_BUILD_CLI=ON' },
    diff_real: { type: 'boolean' }, gpu_only: { type: 'boolean' }, layering_ok: { type: 'boolean', description: 'arch-grep APP-CUDA-FREE-OK' },
    writer_real_vpair: { type: 'boolean', description: 'the f2-dir writer writes the REAL vpair region (F1 NA path works), not the zero-vpair shortcut' },
    no_synthetic: { type: 'boolean' }, goldens_green: { type: 'boolean' },
    extract_reproduces_golden: { type: 'boolean', description: 'extract-f2 on the real AADR -> qpadm reproduced golden_fit0 (+ ideally golden_fitNA maxmiss=0.99) within tier' },
    build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'what landed + the end-to-end gate evidence (the reproduced golden numbers) + the AADR path used; for FAIL exactly what blocked it' },
  },
}
const fixer = [
  'You are a senior C++/CUDA engineer building `steppe extract-f2` (M(cli-4)) for steppe. Do NOT commit (the verdict commits). FIRST clean HEAD: ' + CLEAN + ' . READ docs/design/cli-bindings.md §4 first.', STD, '', DEVLOOP, '',
  'THE RAW AADR IS AT (use this for the golden gate): ' + loc.aadr_paths,
  'THE APPROACH (from the scope step): ' + loc.approach, '',
  'BUILD `steppe extract-f2` per cli-bindings.md §4: the subcommand + arg parsing (--geno/--snp/--ind or --prefix, --pops names|auto-top:K|min-n:N, --out DIR, --blgsize, --maxmiss, --maf, --auto-only, --precision, --device, --dry-run), wire io->decode_af->assign_blocks->compute_f2_blocks_multigpu_device(/_tiered) on the GPU, and the NEW STPF2BK1 f2-dir WRITER (write f2.bin with the REAL vpair from the precompute [F2BlockTensor.vpair], pops.txt in index order, meta.json provenance). app stays a PLAIN CXX target (NO CUDA header — arch-grep). GPU-only (no --device cpu). Add the end-to-end test (tests/cli/, wired into tests/CMakeLists.txt): extract-f2 on the real AADR for the golden_fit0 pop set -> qpadm -> reproduce golden_fit0 within tier (real data, no synthetic). Verify any CUDA/CLI11 API claim vs the docs. Build + ctest + arch-grep until clean+green. Report every file added/changed, the writer design (real vpair), the end-to-end test + the reproduced golden numbers, and the FULL build/ctest/arch-grep output. Do NOT commit. If blocked, report exactly what — do NOT fake the gate or synthesize data.',
].join('\n')
const fix = await tryAgent(fixer, { label: 'fix:extract-f2', phase: 'Build extract-f2' })
if (fix === null) { log('--- extract-f2 fixer died — HALT'); return { halted: true, reason: 'fixer died' } }

await tryAgent(['You are BUILD-REPAIR for M(cli-4) extract-f2. The fixer accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON), patching ONLY trivial -Werror (unused param/var; [[maybe_unused]] for STEPPE_ASSERT-only params; a missing include). DO: ' + RSYNC + ' then ' + BUILD + ' . If a trivial error fires, fix minimally + rebuild, LOOP up to 4x. Do NOT change logic, do NOT revert. If it fails for a NON-trivial reason (a real API/design error, an AADR-read failure, a layering violation), STOP + report. Report final build + the trivial patches.', STD].join('\n'), { label: 'repair:extract-f2', phase: 'Build extract-f2' })

const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for M(cli-4) `steppe extract-f2` (adversarial). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
  'THE RAW AADR: ' + loc.aadr_paths,
  '', 'DO: (1) cd ' + R + ' && git --no-pager diff --stat && git --no-pager diff — confirm real extract-f2 per cli-bindings.md §4 (io->decode->precompute->the NEW STPF2BK1 writer with REAL vpair, not zeros), GPU-only (no --device cpu), app CUDA-free, NO synthetic. (2) RE-RUN: ' + BUILD + ' ; ' + ARCHGREP + ' . (3) THE END-TO-END GOLDEN GATE on the box (REAL AADR): run the built `steppe extract-f2` on ' + loc.aadr_paths + ' for the golden_fit0 pop set (blgsize=0.05, maxmiss=0) -> a fresh f2 dir -> `steppe qpadm --f2-dir <dir> --target England_BellBeaker --left Czechia_EBA_CordedWare,Turkey_N --right <golden_fit0 right>` and confirm it reproduces golden_fit0 weights/p/chisq within the existing tiers; spot-check the extracted f2 matches the steppe precompute / the committed f2_fit0 fixture. (4) PASS only if ALL: real + GPU-only + layering_ok + writer emits REAL vpair; existing ctest green; the end-to-end test passes (golden reproduced THROUGH extract-f2->qpadm on the GPU, real AADR); Release build clean. ',
  'ON PASS: cd ' + R + ' && git add ONLY the M(cli-4) files (src/app extract-f2 + the writer, the new test, the CMake; NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (M(cli-4) extract-f2 + the end-to-end golden reproduction evidence) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the short hash.',
  'ON FAIL: ' + CLEAN + ' (leave repo green) + report exactly what blocked it. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:extract-f2', phase: 'Build extract-f2' })

if (verdict === null) { log('--- extract-f2 verdict died — HALT'); return { halted: true, reason: 'verdict died' } }
if (verdict.pass) log('+++ M(cli-4) extract-f2 committed ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- M(cli-4) extract-f2 FAILED (' + verdict.note + ') — reverted')
return { verdict, aadr: loc.aadr_paths }
