export const meta = {
  name: 'reproduce-another-study',
  description: 'Reproduce ANOTHER published qpAdm study on steppe (after Haak 2015): pick a canonical qpAdm paper with a clean published model + weights, map its populations to the v66 AADR labels (investigating cross-version label SWITCHES/renames), run extract-f2 + qpadm on the 1240K, and compare steppe to the published values. SINGLE-GPU, REAL v66 1240K, no-hash default. Phase 1 PICK+SPEC (web research): choose a well-known qpAdm study (NOT Haak 2015; prefer a different region/era — candidates: Olalde 2018 Britain Bell Beaker ~90% steppe replacement, Olalde 2019 Iberia ~40% steppe, Narasimhan 2019 South Asia Steppe+Iran+AASI, Antonio 2019 Rome, Mathieson 2018 SE Europe, Lazaridis 2022 Southern Arc, Damgaard 2018 Central Asia) with: a clearly-published qpAdm model (target, left/sources, right/outgroups) + numeric weights, and pops that plausibly exist in v66. Extract the EXACT model(s) + the published weights + the paper population labels. Phase 2 MAP+RUN+COMPARE (box): grep the v66 1240K .ind 3rd column to MAP each paper label -> a real v66 AADR label, INVESTIGATING version switches (renames/splits/merges between the paper era and v66, e.g. Anatolia_Neolithic->Turkey_N, Yamnaya_Samara->Russia_Samara_EBA_Yamnaya, WHG->a Mesolithic label); flag any pop absent/ambiguous + pick the best v66 equivalent (note the switch). Then extract-f2 (1240K, --blgsize 0.05 --auto-only; maxmiss 0, relax to 0.5 if singleton-outgroup SNP collapse) + qpadm the model(s); compare steppe weights to published. Phase 3 SYNTHESIZE: write docs/studies/<study>.md (the citation, the model, the paper->v66 pop MAPPING table with switch notes, the steppe result vs published, the runnable one-liner) + commit. REAL data only; SINGLE-GPU (--device 0; multi-gpu parked); HALT-on-fail; resumable on 529.',
  phases: [ { title: 'Pick + spec the study (web)' }, { title: 'Map pops (version switches) + run + compare' }, { title: 'Synthesize study doc + commit' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const PREFIX = '/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB'
const BIN = 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm; full AT2 parity, Haak 2015 already reproduced (docs/studies/haak2015.md). The CLI: steppe extract-f2 --prefix P --pops a,b,.. --out DIR --device 0 --blgsize 0.05 --maxmiss M --auto-only ; then steppe qpadm --f2-dir DIR --target T --left a,b --right r0,r1,.. [--jackknife 2] --format csv. Binary ' + BIN + '.',
  'DATA: REAL AADR v66 1240K at ' + PREFIX + '.{geno,snp,ind} (1.23M SNPs, 23089 ind, TGENO). Population labels are the .ind 3rd column. The HO panel is at /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB if a pop is HO-only.',
  'GOAL: reproduce a SECOND published qpAdm study (not Haak 2015) end-to-end on steppe + the 1240K, and compare to the paper. The CRUX the user flagged: AADR population labels CHANGE across versions (renames/splits/merges) — the paper used older labels; map carefully to the v66 .ind labels and DOCUMENT every switch.',
  'SINGLE-GPU only (--device 0; multi-gpu PARKED). REAL data only (no synthetic). nvcc -> ' + PATHENV + '. Clear core dumps after box runs. Cite the paper + the supplement table for the published model + weights. No fabrication — if a pop is genuinely absent from v66, say so + pick the closest documented proxy (or drop it) and note the caveat.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

phase('Pick + spec the study (web)')
const SPEC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['study','citation','why','models','paper_labels','published'],
  properties: {
    study: { type: 'string', description: 'short name, e.g. "Olalde 2018 (Bell Beaker, Britain)"' },
    citation: { type: 'string', description: 'full citation + the supplement table the model/weights come from' },
    why: { type: 'string', description: 'why this is cleanly reproducible on v66 (clear model, pops likely present, different region/era from Haak)' },
    models: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['name','target','left','right'], properties: {
      name: { type: 'string' }, target: { type: 'string', description: 'paper label' },
      left: { type: 'array', items: { type: 'string' }, description: 'source/left paper labels' },
      right: { type: 'array', items: { type: 'string' }, description: 'outgroup/right paper labels' } } },
      description: '1-3 headline qpAdm models from the paper, in the PAPER population labels' },
    paper_labels: { type: 'array', items: { type: 'string' }, description: 'all distinct paper population labels used (for the mapping step)' },
    published: { type: 'string', description: 'the published weights / p-values for the model(s) (the comparison target)' },
  },
}
const spec = await tryAgent([
  'You are an archaeogenetics researcher. Pick ONE canonical published qpAdm study (NOT Haak 2015) that is cleanly reproducible on the AADR v66 1240K, and extract its headline qpAdm model(s) + published weights. Prefer a different region/era from Haak (Central European steppe). Strong candidates: Olalde 2018 (Britain Bell Beaker ~90% steppe replacement), Olalde 2019 (Iberia ~40% steppe), Narasimhan 2019 (South Asia: Steppe_MLBA + Iran-related + AASI), Antonio 2019 (Rome), Mathieson 2018 (SE Europe), Lazaridis 2022 (Southern Arc), Damgaard 2018 (Central Asia). Use the web (the paper + its supplement) to get the EXACT model (target / left sources / right outgroups) and the PUBLISHED weights+p. Pick one whose populations plausibly exist in v66.', STD,
  '', 'Use WebSearch/WebFetch. Return the structured spec: the study, citation+table, why-reproducible, the model(s) in PAPER labels, all paper labels, and the published weights. Be exact about the model (a wrong source set will not reproduce). No fabrication.',
].join('\n'), { schema: SPEC_SCHEMA, label: 'pick:study', phase: 'Pick + spec the study (web)' })
if (!spec) { log('HALT: study pick failed'); return { halted: true } }
log('study: ' + spec.study + ' — models: ' + spec.models.map(m=>m.name).join(', '))

phase('Map pops (version switches) + run + compare')
const RUN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['done','pop_mapping','version_switches','absent','steppe_results','vs_published','one_liner','note'],
  properties: {
    done: { type: 'boolean' },
    pop_mapping: { type: 'string', description: 'table: paper label -> v66 1240K .ind label (verified present), #individuals each' },
    version_switches: { type: 'string', description: 'the cross-version renames/splits/merges found (paper label vs v66 label) + how resolved' },
    absent: { type: 'string', description: 'any paper pop NOT in v66 + the proxy chosen or the drop + caveat' },
    steppe_results: { type: 'string', description: 'the steppe qpadm weights/p for each model on the 1240K (the actual run output)' },
    vs_published: { type: 'string', description: 'steppe vs the published weights — match? differences + likely cause (label proxy, SNP set, version)' },
    one_liner: { type: 'string', description: 'the verified copy-paste ssh one-liner that reproduces this study (extract-f2 + qpadm), like the Haak one-liner' },
    note: { type: 'string', description: 'maxmiss used + why; any SNP collapse; single-GPU/real confirmed; surprises' },
  },
}
const run = await tryAgent([
  'You are reproducing the study on the box. The picked study + model (PAPER labels):\n' + JSON.stringify(spec, null, 1), STD, '',
  'STEP 1 MAP + investigate version switches: ' + SSH + ' and grep the v66 1240K .ind 3rd column (awk on ' + PREFIX + '.ind) to find the real v66 label for EACH paper label. AADR renames across versions — map carefully (e.g. Anatolia_Neolithic->Turkey_N, Yamnaya_Samara->Russia_Samara_EBA_Yamnaya, WHG->a Mesolithic label, EHG->Russia_*_HG). Record each paper->v66 mapping + the #individuals (so you know singletons), and DOCUMENT every switch. If a pop is absent, pick the closest documented v66 proxy (or drop it) + note the caveat. STEP 2 RUN: ' + BIN + ' extract-f2 --prefix ' + PREFIX + ' --pops <the v66-mapped union> --out /workspace/data/study/f2 --device 0 --blgsize 0.05 --maxmiss 0 --auto-only (if --maxmiss 0 collapses the SNP set because of singleton outgroups, relax to 0.5 and note it); then ' + BIN + ' qpadm for each model (--jackknife 2 --format csv). STEP 3 COMPARE steppe vs the published weights. Clean up the big f2 dir + cores after. Return the structured result incl. a VERIFIED copy-paste ssh one-liner (env + extract-f2 + the qpadm fits) that reproduces it. REAL data; single-GPU.',
].join('\n'), { schema: RUN_SCHEMA, label: 'map-run:study', phase: 'Map pops (version switches) + run + compare' })
if (!run || !run.done) { log('HALT: map/run failed — ' + (run ? run.note : 'agent died')); return { halted: true, spec, run } }
log('ran ' + spec.study + ' — vs published: ' + String(run.vs_published).slice(0,140))

phase('Synthesize study doc + commit')
const verdict = await tryAgent([
  'You are the synthesizer + committer for the reproduced study. Spec:\n' + JSON.stringify(spec,null,1) + '\n\nRun result:\n' + JSON.stringify(run,null,1), STD, '',
  'SANITY-CHECK first (do NOT just trust the run agent): cd ' + R + ' and confirm the one-liner is well-formed; if quick, re-run the qpadm part on the box to confirm the headline weights reproduce. Then WRITE ' + R + '/docs/studies/<study-slug>.md with: (1) the study + citation + the supplement table; (2) the qpAdm model(s); (3) the PAPER->v66 population MAPPING table WITH the version-switch notes (the renames/splits the user wanted documented) + any absent/proxy caveats; (4) the steppe result (weights/se/p per model) vs the PUBLISHED values, with a clear match/differ verdict + likely cause; (5) the verified copy-paste ssh one-liner. Mark REAL v66 1240K / single-GPU / the maxmiss used. Then cd ' + R + ' && git add ONLY docs/studies/<slug>.md (NEVER git add dot; never aadr/), commit with a ROADMAP §6 message (docs(studies): reproduce <study> on steppe + 1240K — pop mapping incl. version switches, steppe vs published) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Return the study name, the headline steppe-vs-published result, the version switches found, and the commit hash + the one-liner.',
].join('\n'), { label: 'synthesize:studydoc', phase: 'Synthesize study doc + commit' })
if (verdict === null) { log('--- synthesize died — HALT'); return { halted: true, spec, run } }
log('+++ study reproduced: ' + String(verdict).slice(0, 240))
return { spec, run, verdict }
