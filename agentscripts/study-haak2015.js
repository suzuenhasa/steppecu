export const meta = {
  name: 'study-haak2015',
  description: 'Reproduce the canonical Haak et al. 2015 ("Massive migration from the steppe") qpAdm story on real AADR v66, end-to-end through the steppe CLI, validated against AT2-on-convertf-PA (the correct reference). steppe reads v66 TGENO correctly; AT2 reads the convertf-converted PACKEDANCESTRYMAP. The headline models: (1) Corded Ware = Yamnaya + Anatolia_Neolithic (THE ~75%-steppe result); (2) European targets (Bell Beaker / Unetice / a modern Euro) = WHG + Anatolia_N + Yamnaya (the 3-way WHG/EEF/Steppe model); (3) Bell Beaker 2-way. Phase 1 SCOPE (read-only, box): grep the v66 .ind for the Haak pops + their individual counts, pick HIGH-N labels (the user wants well-sampled pops), map the canonical models to real v66 labels + a standard Haak-style outgroup/right set. Phase 2 RUN (box): steppe `extract-f2` the union pop set from raw v66 TGENO -> f2 dir -> `steppe qpadm` each model (GPU); AND AT2 extract_f2 on /workspace/data/aadr/converted_pa/v66_HO_pa for the same pops -> qpadm each model (the correct reference). Phase 3 REPORT: per-model weights/se/z/p/feasible from steppe vs AT2-on-convertf (parity = the win, per the user) + whether the Haak biological story is recovered. REAL DATA ONLY (no synthetic); GPU-only for steppe; do NOT modify the steppe repo / commit. Writes /tmp/haak2015-results.md.',
  phases: [ { title: 'Scope (v66 labels + models)' }, { title: 'Run (steppe + AT2)' }, { title: 'Report' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const PATHENV = 'export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
const CORECLEAN = 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'
const OUT = '/tmp/haak2015-results.md'
const RAW = '/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB'           // steppe reads this TGENO directly
const PA  = '/workspace/data/aadr/converted_pa/v66_HO_pa'                 // AT2 reads this (convertf-PA)

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm. The CLI works: `steppe extract-f2 --prefix <geno prefix> --pops <names> --out <dir> [--blgsize 0.05 --maxmiss 0 --autosomes-only]` (genotypes -> f2 dir, GPU, reads v66 TGENO CORRECTLY), then `steppe qpadm --f2-dir <dir> --target T --left a,b[,c] --right r0,r1,.. [--format csv|json]`. The built binary is /workspace/steppe/build-rel/bin/steppe (build with ' + SSH + " 'cd /workspace/steppe && " + PATHENV + " && cmake -S . -B build-rel -GNinja -DCMAKE_BUILD_TYPE=Release -DSTEPPE_BUILD_CLI=ON && cmake --build build-rel --target steppe' if not current). Run it: prefix LD_LIBRARY_PATH=/usr/local/cuda/lib64.",
  'THE DATA (box5090, REAL AADR v66): raw v66 TGENO at ' + RAW + '.{geno,snp,ind} (steppe reads this); convertf-converted PACKEDANCESTRYMAP at ' + PA + '.{geno,snp,ind} (AT2 reads this — AT2 v2.0.10 CANNOT read raw TGENO, memory aadr-tgeno-goldens-corrupt). 27594 inds, 584131 SNPs (HO panel). R + admixtools 2.0.10 installed.',
  'THE STUDY: reproduce Haak et al. 2015 (Massive migration from the steppe). Canonical qpAdm-style results to recover: (a) Corded Ware derives the majority (~70-80%) of its ancestry from Yamnaya/Steppe (the headline); (b) European populations model as WHG + Early-European-Farmer(Anatolia_Neolithic) + Yamnaya/Steppe (the 3-way). VALIDATION BAR (user): steppe must MATCH AT2-on-convertf-PA on the SAME data (parity = the win); exact match to the 2015 PAPER is NOT expected (different AADR vintage; Haak used allsnps/per-quartet — steppe + this AT2 run use maxmiss=0 global-intersection, so both are apples-to-apples). The biological STORY recovering is the bonus.',
  'REAL DATA ONLY — no synthetic. GPU-only for steppe. Do NOT modify the steppe repo, do NOT commit. Long box steps (extract_f2) DETACH + poll. Core dumps: clear with ' + CORECLEAN + ' after builds. Deliverable /tmp/haak2015-results.md.',
].join('\n')

// 1) SCOPE — find the real v66 labels + ind counts, define the models.
phase('Scope (v66 labels + models)')
const SCOPE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['union_pops','models','outgroups','note'],
  properties: {
    union_pops: { type: 'array', items: { type: 'string' }, description: 'the full union of v66 pop labels used across all models (targets+sources+outgroups), each VERIFIED present in the .ind with its ind count' },
    outgroups: { type: 'array', items: { type: 'string' }, description: 'the Haak-style right/outgroup set (v66 labels, present in .ind)' },
    models: { type: 'array', description: 'the Haak models to fit', items: {
      type: 'object', additionalProperties: false, required: ['name','target','left','expect'],
      properties: { name: { type: 'string' }, target: { type: 'string' }, left: { type: 'array', items: { type: 'string' } }, expect: { type: 'string', description: 'the Haak-2015 expectation (e.g. ~0.75 Yamnaya)' } } } },
    note: { type: 'string', description: 'the v66 label mapping + ind counts + any pop that had to be substituted (v66 differences)' },
  },
}
const scope = await tryAgentScope([
  'You are scoping the Haak 2015 reproduction on real AADR v66. READ-ONLY on the box (' + SSH + '). grep the .ind (col 3 = pop label) of ' + RAW + '.ind for the Haak-relevant populations + COUNT individuals per label (e.g. `awk \047{print $3}\047 ' + RAW + '.ind | sort | uniq -c | sort -rn`), and pick HIGH-N, well-sampled v66 labels for each role:', STD, '',
  'ROLES (map to the best-sampled v66 labels actually present): TARGETS — Corded Ware (e.g. Czechia_EBA_CordedWare / Germany_CordedWare* / a high-N CordedWare), plus 1-2 European targets (Bell_Beaker, Unetice, or a modern European e.g. French/Sardinian) for the 3-way; SOURCES — Yamnaya/Steppe (Russia_Samara_EBA_Yamnaya or another high-N Yamnaya), Anatolia_Neolithic (Turkey_N / Anatolia_N / a high-N early farmer = EEF), WHG (Loschbour / WHG / Iron_Gates_HG / a high-N western hunter-gatherer); OUTGROUPS/RIGHT — a standard Haak/qpAdm right set from v66 (e.g. Mbuti, Ust_Ishim, Kostenki14, MA1/Mal\047ta(Russia_MA1), Han, Papuan, Onge, Karitiana, Natufian/Israel_Natufian, Iran_GanjDareh_N) — pick those PRESENT with decent N.',
  'DEFINE the models: (1) "CordedWare = Yamnaya + Anatolia_N" (expect ~0.7-0.8 Yamnaya); (2) "BellBeaker = Yamnaya + Anatolia_N" (or CordedWare-derived); (3) a 3-way "EuropeanTarget = WHG + Anatolia_N + Yamnaya". Use ONE shared outgroup/right set (>=6, more than #sources). Return the structured scope: the union pop set, the right set, the models, and the label mapping + ind counts. Confirm EVERY label is in the .ind. NO guessing — verify against the .ind.',
].join('\n'), { schema: SCOPE_SCHEMA, label: 'scope:haak', phase: 'Scope (v66 labels + models)' })

if (!scope || !scope.models || scope.models.length === 0) { log('HALT: scope failed — ' + (scope ? scope.note : 'agent died')); return { halted: true, scope } }
log('Haak models scoped: ' + scope.models.map(m => m.name).join(', ') + ' | union ' + scope.union_pops.length + ' pops, ' + scope.outgroups.length + ' outgroups')

// 2) RUN — steppe (extract-f2 -> qpadm) AND AT2-on-convertf, same models.
phase('Run (steppe + AT2)')
const runShared = 'THE SCOPE:\nunion pops (' + scope.union_pops.length + '): ' + scope.union_pops.join(',') + '\noutgroups: ' + scope.outgroups.join(',') + '\nmodels:\n' + scope.models.map(m => '  - ' + m.name + ': target=' + m.target + ' left=[' + m.left.join(',') + '] right=[outgroups] (Haak expect: ' + m.expect + ')').join('\n')

const steppeRun = await tryAgentRun([
  'You are running the Haak 2015 models through the STEPPE CLI on the GPU (box5090). Build the f2 ONCE over the union pop set from the raw v66 TGENO, then qpAdm each model.', STD, '', runShared, '',
  'DO: (1) ensure the steppe binary is current (build if needed, ' + SSH + ', Release -DSTEPPE_BUILD_CLI=ON, target steppe; ' + CORECLEAN + ' after). (2) `steppe extract-f2 --prefix ' + RAW + ' --pops <union,comma-sep> --out /workspace/data/haak/steppe_f2 --blgsize 0.05 --maxmiss 0 --autosomes-only` (GPU; reads v66 TGENO; DETACH+poll if long). (3) for EACH model: `LD_LIBRARY_PATH=/usr/local/cuda/lib64 /workspace/steppe/build-rel/bin/steppe qpadm --f2-dir /workspace/data/haak/steppe_f2 --target <T> --left <a,b[,c]> --right <outgroups,comma-sep> --format json` and capture weights/se/z/p/chisq/dof/f4rank/feasible + the SNP count. Report each model\047s full result. REAL DATA; GPU only.',
].join('\n'), { label: 'run:steppe', phase: 'Run (steppe + AT2)' })

const at2Run = await tryAgentRun([
  'You are running the SAME Haak 2015 models through ADMIXTOOLS 2 on the convertf-converted PACKEDANCESTRYMAP (the CORRECT reference; AT2 cannot read raw v66 TGENO). box5090, R + admixtools 2.0.10.', STD, '', runShared, '',
  'DO: in R (Rscript, DETACH+poll): `extract_f2("' + PA + '", "/workspace/data/haak/at2_f2", pops=c(<union>), blgsize=0.05, maxmiss=0, overwrite=TRUE, n_cores=8)` then for EACH model `qpadm("/workspace/data/haak/at2_f2", target=<T>, left=c(<a,b[,c]>), right=c(<outgroups>), boot=FALSE)` and capture res$weights (est/se/z), the p-value/rankdrop, the SNP count. Report each model\047s full AT2 result. REAL DATA only; this is the correctness reference steppe is compared against.',
].join('\n'), { label: 'run:at2', phase: 'Run (steppe + AT2)' })

// 3) REPORT
phase('Report')
const report = await tryAgentRun([
  'You are writing the Haak 2015 reproduction report for steppe. Compare the steppe-CLI results vs AT2-on-convertf-PA (the correct reference), per model.', STD, '', runShared, '',
  'STEPPE RESULTS:\n' + String(steppeRun).slice(0, 6000), '', 'AT2-ON-CONVERTF RESULTS:\n' + String(at2Run).slice(0, 6000), '',
  'WRITE ' + OUT + ' with: (1) the model table per model: target | sources | steppe weights[se]/p/feasible | AT2-on-convertf weights[se]/p/feasible | MATCH? (steppe vs AT2 within tier — the user\047s win condition) | the Haak-2015 expectation. (2) The HEADLINE: did steppe recover the Haak story — Corded Ware majority-Yamnaya (~70-80%)? the 3-way WHG/EEF/Steppe European model feasible? (3) steppe-vs-AT2 PARITY verdict (the residual block-partition ~0.3% gap is expected; flag if anything is larger). (4) the SNP counts + any caveat (maxmiss=0 vs Haak allsnps). Use Write. Return the headline + the per-model steppe-vs-AT2 match at a glance.',
].join('\n'), { label: 'report:haak', phase: 'Report' })
log('Haak 2015 study -> /tmp/haak2015-results.md: ' + String(report).slice(0, 200))
return { scope, steppeRun, at2Run, report }

async function tryAgentScope(p, opts) { let r = await agent(p, opts); if (r === null) { r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }
async function tryAgentRun(p, opts) { let r = await agent(p, opts); if (r === null) { r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }
