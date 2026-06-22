export const meta = {
  name: 'ancestry-atlas-survey',
  description: 'The big kickass showcase: a CURATED, large-scale qpAdm ANCESTRY ATLAS — systematically model EVERY ancient West-Eurasian population as 1-3-way mixtures of the canonical ancestral sources, WITH standard errors, on real v66 1240K, single-GPU. This is impractical on AT2 (100k+ fits with SE = weeks) but minutes on steppe (batched rotation ~60k models/sec jk0 / ~20k jk1, the streaming f2 cache, the jk1 SE-budget fix). NOT the alphabetical-first-100 noise — a CURATED source pool + a STRONG outgroup set so the feasible models are INTERPRETABLE. Phase 1 DESIGN (web + box .ind): choose (a) a CURATED ~12-20 source pool of canonical, REAL-sampled v66 ancestral sources (WHG e.g. Luxembourg_Loschbour_Mesolithic; EHG e.g. Russia_Karelia_Mesolithic; CHG e.g. Georgia_Kotias; Iran_N e.g. Iran_GanjDareh_N; Anatolia_N=Turkey_N; Levant e.g. Israel_Natufian/Israel_PPNB; Steppe e.g. Russia_Samara_EBA_Yamnaya; + Caucasus/Siberia/etc.) — NO ghosts/clines, all verified present in the v66 1240K .ind; (b) a STRONG distal ~9-pop outgroup/right set (the Reich base set: Mbuti, Russia_UstIshim_IUP, Russia_Kostenki_UP/Russia_MA1, Han, Papuan, Karitiana, an Onge proxy e.g. Atayal/Dai, Ethiopia_MotaCave_4500BP, etc. — verified, distinct from sources); (c) a coherent TARGET set of ancient West-Eurasian populations (Neolithic->Iron Age Europe + Anatolia + Steppe + Near East), >=8 individuals, autosomal, ~100-200 targets, mapped to v66 labels (note version switches). State the science framing (the ancestry of ancient West Eurasia as WHG/EHG/CHG/Iran/Anatolia/Levant/Steppe combinations). Phase 2 RUN (box, single-GPU): build ONE f2 cache over {all targets + the source pool + the right set}; loop qpadm-rotate per target over the source pool (--min-sources 1 --max-sources 3 --jackknife 1) -> collect per target the BEST feasible model (parsimony: fewest sources, then highest p, p>0.05); record the total models fit + wall. Phase 3 SYNTHESIZE: the ATLAS table (target -> best-fit model, weights, SE, p, #sources), the headline patterns (e.g. the steppe-ancestry gradient across Bronze Age Europe; EEF/HG gradients), the scale (N targets x M models = total fits with SE, in T minutes; vs the AT2 estimate = weeks), write docs/studies/ancestry-atlas.md + commit. REAL v66 1240K only; SINGLE-GPU (--device 0; multi-gpu parked); curated (no ghosts); HALT-on-fail; resumable on 529.',
  phases: [ { title: 'Design the curated survey (sources/right/targets)' }, { title: 'Build f2 cache + rotate every target' }, { title: 'Synthesize the ancestry atlas + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. main @ 8babe0f. The CLI (single-GPU --device 0): steppe extract-f2 --prefix P --pops a,b,.. --out DIR --device 0 --blgsize 0.05 --maxmiss M --auto-only ; steppe qpadm-rotate --f2-dir DIR --target T --pool p1,.. --right r1,.. --min-sources 1 --max-sources 3 --jackknife 1 --format csv --out FILE --device 0. The rotation engine is batched (~60k models/sec jk0, ~20k jk1) + the jk1 SE-pass VRAM budget is fixed (SEs scale single-GPU). Binary ' + BIN + '.',
  'DATA: REAL AADR v66 1240K at ' + PREFIX + '.{geno,snp,ind} (1.23M SNPs, 23089 ind, 926 pops >=5 ind). Population labels = the .ind 3rd column (awk "{print $3}").',
  'GOAL: a CURATED large-scale qpAdm ancestry ATLAS — model every ancient West-Eurasian target as 1-3-way mixtures of the canonical ancestral sources, WITH SEs. The 100-random-pop run produced only noise (uncurated pool + thin outgroups -> spurious overfits); THIS must be curated: a canonical source pool (NO ghosts/clines — all real sampled v66 pops) + a STRONG distal outgroup set, so feasible models are interpretable. The showcase: 100k+ qpAdm fits WITH SE in minutes (vs AT2 weeks).',
  'SINGLE-GPU only (--device 0; multi-gpu PARKED). REAL v66 only (no synthetic). nvcc -> ' + PATHENV + '. Clear core dumps. Cite v66 .ind labels (verify present + individual counts). No fabrication — if a canonical source has no clean v66 pop, pick the documented proxy + note it.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Design the curated survey (sources/right/targets)')
const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','source_pool','right_set','targets','framing','notes'],
  properties: {
    done: { type: 'boolean' },
    source_pool: { type: 'string', description: 'the CURATED ~12-20 canonical ancestral sources -> v66 1240K .ind labels (verified present, #ind each); the ancestry each represents (WHG/EHG/CHG/Iran/Anatolia/Levant/Steppe/...); no ghosts' },
    right_set: { type: 'string', description: 'the STRONG distal ~9-pop outgroup set -> v66 labels (verified, distinct from sources)' },
    targets: { type: 'string', description: 'the coherent ancient West-Eurasian TARGET set -> v66 labels (>=8 ind, autosomal, ~100-200 pops); how chosen + the comma-list (or the awk rule to derive it on the box)' },
    framing: { type: 'string', description: 'the science framing + why this is the kickass-at-scale showcase (the model count, vs AT2)' },
    notes: { type: 'string', description: 'version switches / proxies; any caveats' },
  },
}
const design = await tryAgent([
  'You are an archaeogenetics methods expert designing a CURATED large-scale qpAdm ancestry atlas over the v66 1240K. Choose the curated source pool, the strong outgroup set, and the ancient West-Eurasian target set — all mapped to + VERIFIED present in the v66 1240K .ind (' + SSH + ', awk the 3rd column for labels + counts).', STD, '',
  'DELIVER: (a) the CURATED source pool (~12-20 canonical W.Eurasian ancestral sources: WHG/EHG/CHG/Iran_N/Anatolia_N/Levant_N/Steppe + Caucasus/Siberia/etc., each a REAL sampled v66 pop, NO ghosts/clines, verified present + #ind); (b) the STRONG distal outgroup/right set (~9, the Reich base set, verified, distinct from the sources); (c) the TARGET set — coherent ancient West-Eurasian pops (Neolithic->Iron Age Europe + Anatolia + Steppe + Near East), >=8 individuals, autosomal, ~100-200 targets (give the explicit list OR a precise awk rule to derive it from the .ind, e.g. by region/period keywords + count threshold); (d) the science framing. Verify EVERYTHING against the box .ind. Return the structured design. No ghosts, no fabrication.',
].join('\n'), { schema: DESIGN_SCHEMA, label: 'design:survey', phase: 'Design the curated survey (sources/right/targets)' })
if (!design || !design.done) { log('HALT: survey design failed'); return { halted: true, design } }
log('survey: ' + String(design.framing).slice(0,120))

phase('Build f2 cache + rotate every target')
const RUN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','cache','atlas','scale','at2_estimate','one_liner','notes'],
  properties: {
    done: { type: 'boolean' },
    cache: { type: 'string', description: 'the f2 cache: #pops (targets+sources+right), SNPs kept, blocks, build time, tier, f2.bin size' },
    atlas: { type: 'string', description: 'the per-target best-fit table: target -> best feasible model (sources + weights + SE), p, #sources — for a representative/headline subset (the full table goes in the doc); how many targets got a feasible model' },
    scale: { type: 'string', description: 'the scale: #targets, models-per-target, TOTAL model-fits WITH SE, total wall time (cache build + all rotations)' },
    at2_estimate: { type: 'string', description: 'the same survey on AT2: rough estimate (the per-fit AT2 cost x total fits) -> the speedup, the could-not-be-done-before claim' },
    one_liner: { type: 'string', description: 'a verified copy-paste ssh command that reproduces the survey (cache + the per-target rotation loop)' },
    notes: { type: 'string', description: 'maxmiss; #feasible per target distribution; single-GPU/real confirmed; any target that failed' },
  },
}
const run = await tryAgent([
  'You are running the curated ancestry atlas on the box. The design:\n' + JSON.stringify(design, null, 1), STD, '',
  'STEP 1: build ONE f2 cache over {all targets + the source pool + the right set} on the v66 1240K (' + BIN + ' extract-f2 --device 0 --blgsize 0.05 --maxmiss 0.5 --auto-only; streamed tier auto if big; record build time/SNPs/blocks/tier/size). STEP 2: loop qpadm-rotate over EACH target: ' + BIN + ' qpadm-rotate --f2-dir CACHE --target <T> --pool <the curated source pool> --right <the right set> --min-sources 1 --max-sources 3 --jackknife 1 --device 0 --format csv --out /tmp/atlas/<T>.csv ; from each, pick the BEST feasible model (feasible==TRUE & p>0.05; parsimony: fewest sources, then highest p). STEP 3: assemble the atlas (target -> best model + weights + SE + p) + the SCALE (total model-fits = #targets x models-per-target, total wall) + an AT2 estimate (per-fit cost x total). Clean the cache + per-target csvs (keep a summary) + cores after. Return the structured result incl a verified copy-paste one-liner. REAL data, single-GPU. If many targets have NO feasible model, note it (may need to relax to max-sources 4 or revisit the source pool).',
].join('\n'), { schema: RUN_SCHEMA, label: 'run:atlas', phase: 'Build f2 cache + rotate every target' })
if (!run || !run.done) { log('HALT: atlas run failed — ' + (run ? run.notes : 'agent died')); return { halted: true, design, run } }
log('atlas: ' + String(run.scale).slice(0,140))

phase('Synthesize the ancestry atlas + commit')
const verdict = await tryAgent([
  'You are the synthesizer + committer for the ancestry atlas. Design:\n' + JSON.stringify(design,null,1) + '\n\nRun:\n' + JSON.stringify(run,null,1), STD, '',
  'SANITY-CHECK (do NOT just trust the run agent): spot-verify 2-3 headline targets reproduce a sensible, KNOWN result (e.g. a Bronze Age European target shows substantial Steppe/Yamnaya; a Neolithic European shows mostly Anatolia_N+WHG; a Near Eastern target shows Iran/Levant) — these are well-established, so a wildly wrong call means a setup bug. Then WRITE ' + R + '/docs/studies/ancestry-atlas.md: (1) the framing + the curated source pool / right set / target set (with v66 labels + version-switch notes); (2) the ATLAS table (target -> best-fit model, weights, SE, p, #sources) — the full set or a clearly-sampled representative set with the count; (3) the headline PATTERNS (the steppe gradient across BA Europe, the EEF/HG clines, Near-East Iran/Levant) cross-checked against known archaeogenetics; (4) THE SCALE: total qpAdm fits WITH SE, the wall time, and the AT2 estimate -> the "could not be done before, can now" headline (be honest about the AT2 estimate basis). Mark REAL v66 1240K / single-GPU / curated. Then cd ' + R + ' && git add ONLY docs/studies/ancestry-atlas.md (NEVER git add dot; never aadr/), commit with a ROADMAP §6 message (docs(studies): curated large-scale qpAdm ancestry atlas on v66 1240K — N targets x curated sources, X fits with SE in T min single-GPU) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Return the headline: #targets, total fits, wall, the standout patterns, + the commit hash.',
].join('\n'), { label: 'synthesize:atlas', phase: 'Synthesize the ancestry atlas + commit' })
if (verdict === null) { log('--- synthesize died — HALT'); return { halted: true, design, run } }
log('+++ ancestry atlas: ' + String(verdict).slice(0, 240))
return { design, run, verdict }
