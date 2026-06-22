export const meta = {
  name: 'fix-extract-f2-hash',
  description: 'Fix the extract-f2 ~40s bottleneck: it spends ~37.5s of ~41s on a byte-at-a-time scalar SHA-256 of the entire 6.7GB source .geno (provenance hash) — sha256_of_file at src/app/f2_dir_writer.cpp:256 (Sha256::update byte-at-a-time :37-42,134-144), called from cmd_extract_f2.cpp:453; proven CPU-bound (page-cached, read_bytes=0; 190MB/s scalar-SHA vs sha256sum SHA-NI 1.29GB/s). THE CHANGE (user-approved): (1) make NO-HASH the DEFAULT — extract-f2 does NOT compute the source-geno provenance hash by default (-> ~2-3s); add an OPT-IN flag --hash (default off) to request it; meta.json records the source-hash as intentionally skipped (e.g. geno_sha256 empty/null + source_hash_computed:false) so a consumer knows the absence is deliberate, not corruption. (2) when --hash IS given, make it FAST + OVERLAPPED: replace the byte-at-a-time SHA with a bulk/block-based SHA-256 (prefer a correct portable block impl with NO new heavy dep; use SHA-NI/OpenSSL only if cleanly already-available), AND overlap the hash on a background std::thread with the decode+gather+f2-GPU work (the hash only needs the .geno path) so --hash wall ~= max(hash, f2) instead of additive (~5.5s not ~40s). Keep PROVENANCE value (it caught the corrupt qpwave golden). GOLDEN-GATE: ensure tests (esp. cli_extract_qpadm) stay green — if a test asserts geno_sha256, run that extract-f2 with --hash or update it; do NOT weaken the fit goldens. Then RE-MEASURE on 1240K single-GPU (--device 0, Release): extract-f2 DEFAULT (no-hash) wall (~2-3s target) + --hash wall, and CORRECT docs/perf/1240k-sweep.md (the bottleneck was the provenance SHA, NOT decode/IO). fixer -> build-repair -> verdict(+re-measure+doc+commit); HALT-on-fail; SINGLE-GPU (multi-gpu parked); REAL data.',
  phases: [ { title: 'Implement --no-hash default + fast/overlapped --hash' }, { title: 'Golden-gate + re-measure + correct doc + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const RSYNC = 'rsync -az --delete-after --exclude .git --exclude build --exclude build-rel --exclude aadr -e ssh ' + R + '/ box5090:/workspace/steppe/'
const BUILD = SSH + " 'cd /workspace/steppe && " + PATHENV + " && " + CORECLEAN + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON >/tmp/cfg.log 2>&1 && cmake --build build-rel 2>&1 | tail -25 && echo === CTEST === && STEPPE_THOROUGH=1 ctest --test-dir build-rel --output-on-failure 2>&1 | tail -55; " + CORECLEAN + "'"
const CLEAN = 'cd ' + R + ' && git checkout -- . && git clean -fd src tests include docs'
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. Branch phase2-fit-engine == main @ 168c63e. extract-f2 (src/app/cmd_extract_f2.cpp + f2_dir_writer.cpp) builds an f2 dir (f2.bin + pops.txt + meta.json) from a genotype prefix. The CLI: steppe extract-f2 --prefix P --pops a,b,.. --out DIR --blgsize 0.05 --maxmiss 0 [--ploidy auto].',
  'THE BUG: f2_dir_writer computes a SHA-256 of the WHOLE source .geno (provenance, stored as geno_sha256 in meta.json) using a byte-at-a-time Sha256::update -> ~37.5s of ~41s on the 6.7GB 1240K .geno (proven CPU-bound: page-cached, /proc read_bytes=0; 190MB/s scalar vs sha256sum SHA-NI 1.29GB/s). The genuine genotype read (per-individual seek+gather, geno_reader.cpp:194-217) + decode + f2 GEMM are each <1s. So extract-f2 is wasting ~37s on a naive provenance hash.',
  'THE FIX (user-approved): (1) --no-hash is the DEFAULT — do NOT hash the source .geno unless asked; add opt-in --hash (default OFF); meta.json marks the hash intentionally skipped (geno_sha256 empty/null + a source_hash_computed:false marker) so consumers know absence is deliberate. (2) when --hash: a fast SHA (bulk/block-based, prefer NO new heavy dep; SHA-NI/OpenSSL only if cleanly already-linked) AND overlap it on a background std::thread with the decode+gather+f2-GPU work (the hash only needs the .geno path), join before writing meta.json. KEEP the provenance value (it caught the corrupt qpwave golden via the e588406 vs 7af8c2f5 mismatch).',
  'GOLDEN-GATE: the AT2-fixture fit goldens (qpadm_parity/rotation/qpwave) read fixtures, unaffected. The cli_extract_qpadm e2e runs steppe extract-f2 -> if it asserts the meta geno_sha256, either run that extract with --hash or update the assertion to the no-hash marker (do NOT weaken the fit comparison). Full ctest must stay green. NAMING-STYLE-STANDARD + the CLI/meta.json schema (docs/design/cli-bindings.md); verify any threading/std API; no heavy new deps without justification.',
  'REAL DATA; SINGLE-GPU only (--device 0; multi-gpu PARKED, memory multi-gpu-parked). Box ' + SSH + '; nvcc -> ' + PATHENV + '; RELEASE -DSTEPPE_BUILD_CLI=ON; nothing builds locally; clear core dumps per build.',
].join('\n')

const DEVLOOP = 'DEV LOOP: clean HEAD (' + CLEAN + '). Edit locally; rsync (' + RSYNC + '); build+ctest (' + BUILD + '). Do NOT commit (the verdict commits). NO synthetic. SINGLE-GPU.'

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Implement --no-hash default + fast/overlapped --hash')
const fixer = [
  'You are a senior engineer fixing the extract-f2 provenance-hash bottleneck (user-approved: no-hash default + fast/overlapped opt-in --hash). Do NOT commit. Start clean: ' + CLEAN + '. READ src/app/f2_dir_writer.cpp (the Sha256 class + sha256_of_file), src/app/cmd_extract_f2.cpp (where it is called, :453), the CLI flag plumbing (cli_parse.cpp / cli_args.hpp / config_builder.cpp / run_config.hpp), and tests/cli/test_cli_extract_qpadm.cpp.', STD, '', DEVLOOP, '',
  'IMPLEMENT: (1) add the opt-in --hash flag (default OFF) to extract-f2, plumbed CLI->config->cmd_extract_f2; default skips the source-geno SHA entirely (do not even read the .geno for hashing). meta.json: when skipped, geno_sha256 = "" (or null) + a clear source_hash_computed:false marker. (2) when --hash: replace the byte-at-a-time SHA with a bulk/block-based SHA-256 (correct, no new heavy dep preferred; SHA-NI/OpenSSL only if already cleanly available + justified) AND overlap it on a std::thread started BEFORE the decode+f2 GPU pipeline, joined before meta.json is written. Verify the hash value is unchanged (a known .geno hashes to the same digest as before / as sha256sum). Keep cli_extract_qpadm green (use --hash there if it needs the sha, else update to the no-hash marker; do NOT weaken the fit comparison). Build + full STEPPE_THOROUGH ctest. SANITY (no commit): time extract-f2 on the 1240K (single-GPU) WITHOUT --hash (expect ~2-3s) and WITH --hash (expect fast/overlapped). Report every file changed, the SHA approach, the timings, and the FULL ctest. Do NOT commit.',
].join('\n')
const fix = await tryAgent(fixer, { label: 'fix:hash', phase: 'Implement --no-hash default + fast/overlapped --hash' })
if (fix === null) { log('--- fixer died — HALT'); return { halted: true } }

await tryAgent(['BUILD-REPAIR for the extract-f2 hash fix. Accumulated edits (do NOT clean/revert). Reach a CLEAN Release build (-DSTEPPE_BUILD_CLI=ON), patching only trivial -Werror. DO: ' + RSYNC + ' then ' + BUILD + '. Loop up to 4x on trivial errors. NON-trivial -> STOP + report. Report final build + ctest + patches.', STD].join('\n'), { label: 'repair', phase: 'Implement --no-hash default + fast/overlapped --hash' })

phase('Golden-gate + re-measure + correct doc + commit')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['pass','default_no_hash','hash_opt_in_fast','goldens_green','extract_f2_default_sec','extract_f2_hash_sec','provenance_kept','build_clean','commit_hash','note'],
  properties: {
    pass: { type: 'boolean', description: 'true ONLY if: --no-hash is the default (extract-f2 ~2-3s on 1240K single-GPU), --hash is opt-in + fast/overlapped + produces the correct digest, the provenance marker is in meta.json, full STEPPE_THOROUGH ctest green (esp cli_extract_qpadm), Release build clean, single-GPU, no synthetic, no fit-golden weakening' },
    default_no_hash: { type: 'boolean' }, hash_opt_in_fast: { type: 'boolean', description: '--hash is opt-in, fast (bulk SHA) + overlapped, correct digest' },
    goldens_green: { type: 'boolean' },
    extract_f2_default_sec: { type: 'string', description: 'extract-f2 wall on 1240K single-GPU WITHOUT --hash (target ~2-3s, was ~40s)' },
    extract_f2_hash_sec: { type: 'string', description: 'extract-f2 wall WITH --hash (fast+overlapped)' },
    provenance_kept: { type: 'boolean', description: '--hash still yields the correct geno_sha256 (matches sha256sum) so provenance is preserved' },
    build_clean: { type: 'boolean' }, commit_hash: { type: 'string' },
    note: { type: 'string', description: 'the SHA approach + overlap; default vs --hash timings; how cli_extract_qpadm was kept green; for FAIL the exact issue' },
  },
}
const verdict = await tryAgent([
  'You are the INDEPENDENT VERDICT for the extract-f2 hash fix (adversarial). The fixer reported:\n<<<\n' + fix + '\n>>>', STD, '',
  'DO: (1) cd ' + R + ' && git --no-pager diff — confirm --no-hash default + opt-in --hash (fast bulk SHA + std::thread overlap), the meta.json skipped-marker, cli_extract_qpadm kept green WITHOUT weakening the fit comparison, no fit golden touched. (2) ' + BUILD + ' — STEPPE_THOROUGH ctest MUST be green. (3) RE-MEASURE single-GPU (--device 0) on the 1240K: extract-f2 WITHOUT --hash (expect ~2-3s, was ~40s) and WITH --hash (fast+overlapped); confirm --hash yields the SAME digest as sha256sum of the .geno (provenance preserved). PASS only if all hold. ',
  'ON PASS: cd ' + R + ' && git add ONLY the changed source/test/doc files (NEVER git add dot; never aadr/ build_run.sh f2_emu_spike.cu handoff-*.md), commit with a ROADMAP §6 message (extract-f2: no-hash default + opt-in fast/overlapped --hash -> ~40s to ~2-3s; the ~37s was a byte-at-a-time provenance SHA, not decode/IO; provenance preserved under --hash) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . ALSO correct docs/perf/1240k-sweep.md (the extract-f2 ~40s was the provenance SHA-256, NOT decode/IO-bound; default now ~2-3s; --hash ~Xs fast+overlapped; the genuine read+decode+f2 floor is ~2-3s). Capture the hash + the default vs --hash timings.',
  'ON FAIL: ' + CLEAN + ' and report the exact residual. Return the structured verdict.',
].join('\n'), { schema: VERDICT_SCHEMA, label: 'verdict:hash', phase: 'Golden-gate + re-measure + correct doc + commit' })
if (verdict === null) { log('--- verdict died — HALT'); return { halted: true } }
if (verdict.pass) log('+++ extract-f2 ' + verdict.extract_f2_default_sec + ' default (was ~40s) ' + verdict.commit_hash + ' — ' + verdict.note)
else log('--- FAILED (' + verdict.note + ')')
return { verdict }
