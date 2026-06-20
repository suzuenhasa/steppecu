export const meta = {
  name: 'research-desirable-features',
  description: 'Research fan-out: a multi-modal web search sweep on WHICH FEATURES people want in qpAdm / ADMIXTOOLS / population-genetics f-statistics tooling — existing-but-painful AND wished-for-but-missing. Parallel search agents each cover a different angle (admixtools GitHub issues/discussions, popgen forums/biostars/anthrogenica/reddit, papers methods+future-work+limitations, adjacent/competitor tools, performance pain points, the manual rotation/model-competition workflows), then adversarial completeness critic, then a synthesis mapping each desired feature to what steppe HAS (f2 precompute, qpAdm/qpWave/rotation) vs LACKS (standalone f4/f3/D-stat/f4-ratio/qpdstat, qpGraph, DATES, admixture dating, etc.), prioritized by demand x fit-with-steppes-GPU-strength. Cited. No box, no repo edits — returns the synthesized report.',
  phases: [
    { title: 'Sweep', detail: 'parallel multi-angle web searches: GitHub issues, forums, papers, competitor tools, perf pain points, rotation workflows' },
    { title: 'Synthesize', detail: 'adversarial completeness critic + a prioritized, cited feature wishlist mapped to steppe has/lacks' },
  ],
}

const STEPPE = 'steppe = a GPU/CUDA-13/Blackwell reimplementation of ADMIXTOOLS 2 f-statistics + qpAdm. ALREADY BUILT: the f2 precompute (device-resident f2_blocks, streamed, multi-GPU, scales full-autosome ~2500 pops), and the qpAdm fit engine S3-S8 ON THE GPU — the GLS weight fit, the rank test / qpWave (rankdrop/popdrop), arbitrary model sizes via cuSOLVER, and the model-space ROTATION (thousands of models batched + sharded across GPUs, ~5-6k models/sec), all parity-validated vs ADMIXTOOLS 2 goldens on real AADR data. NOT YET exposed: standalone f4 / f3 / D-statistic / f4-ratio / outgroup-f3 / qpdstat entry points (the f4 math exists internally in the fit), qpGraph (admixture graphs), DATES/admixture-dating, a CLI, Python bindings. steppes edge is GPU throughput: precompute-once / fit-many, huge model-space searches, batched rotation.'

const ANGLES = [
  { key: 'admixtools-issues', focus: 'the ADMIXTOOLS 2 R package (uqrmaie1/admixtools) + classic ADMIXTOOLS (DReichLab/AdmixTools): GitHub issues, discussions, the docs/vignette wishlists, and the package NEWS/TODO — what users file as feature requests, bugs, performance complaints, confusions, and "can it do X?" questions. Also the readme/wiki "limitations" / "not implemented".' },
  { key: 'forums-community', focus: 'population-genetics practitioner communities — Biostars, r/genetics, r/AncientDNA, anthrogenica, the eurogenes/vahaduo blog ecosystem, twitter/bluesky popgen — what people struggle with or wish existed in qpAdm/f-stat workflows (speed, batch model testing, rotation automation, reproducibility, visualization, ease of use, model-competition/feasibility ranking).' },
  { key: 'papers-methods', focus: 'recent ancient-DNA / popgen METHODS papers and the methods/limitations/future-work sections of qpAdm-using papers (Lazaridis, Reich lab, Patterson, Harney et al. "Assessing the performance of qpAdm", Narasimhan, Olalde): documented qpAdm limitations, the "rotating outgroups"/competition methodology, recommended diagnostics, and what authors say is missing or slow.' },
  { key: 'adjacent-tools', focus: 'adjacent / competitor / complementary tools and what FEATURES they offer that an f-stat engine could match or integrate: qpGraph / ADMIXTOOLS graph fitting, DATES (admixture dating), Galaxy/qpAdm wrappers, KGD/treemix/qpBrute/admixturegraph, momentsLD/momi2, the vahaduo web qpAdm, plus PCA/ADMIXTURE/F4-ratio/D-stat (qpDstat) usage patterns.' },
  { key: 'performance-pain', focus: 'PERFORMANCE + scale pain points in the qpAdm/f-stat workflow that a GPU engine specifically fixes: extract_f2 time/memory at large pop counts, the cost of large rotations / many-model searches, jackknife re-fits, memory limits, reproducibility/determinism complaints, and where practitioners say "this takes hours/days".' },
  { key: 'workflow-rotation', focus: 'the WORKFLOWS people assemble by hand that could be first-class features: rotating right/left sets and ranking feasible models, automated source competition, nested-model selection, p-value/SE diagnostics, "is my model well-determined", multi-dataset merge/harmonization, QC, and result visualization/reporting/export.' },
]

phase('Sweep')
const findings = await parallel(ANGLES.map(a => () => agent([
  'You are a population-genetics-aware research analyst. Use web search to find WHAT FEATURES people want (existing-but-painful OR wished-for-but-missing) in qpAdm / ADMIXTOOLS / popgen f-statistics tooling, for THIS angle: ' + a.focus,
  '', 'CONTEXT — ' + STEPPE,
  '', 'Load web tools first: ToolSearch with query "select:WebSearch,WebFetch", then run 5-10 targeted searches for this angle and FETCH the most relevant primary sources (GitHub issue threads, forum posts, paper PDFs/abstracts, tool docs). Extract CONCRETE, SOURCED desiderata — each as: the feature/pain, who wants it + evidence (quote/issue#/url), how common it seems, and whether it plays to a GPU throughput engine. Prefer primary sources over blogspam. Do NOT fabricate URLs or quotes — if a claim is inferred, say so.',
  'Return a structured list of 6-15 desiderata for this angle: {feature, the pain/why, evidence+URL, demand signal (anecdotal/recurring/widely-cited), GPU-fit (does steppes precompute-once-fit-many / batched-rotation strength help?)}. Cite real URLs.',
].join('\n'), { label: 'sweep:' + a.key, phase: 'Sweep' })))

phase('Synthesize')
const critic = await agent([
  'You are an adversarial COMPLETENESS critic for a feature-desirability survey of qpAdm/f-stat tooling. Below are 6 angle reports. Identify what is MISSING or under-covered: modalities not searched, obvious features nobody mentioned (e.g. f4/f3/D-stat exposure, qpGraph, DATES, CLI, Python API, visualization, multi-dataset merge, bootstrap, qpfstats heavy-missingness, model auto-search/competition), weak/uncited claims, and duplicate-but-differently-named items. Do 3-6 of your OWN web searches (ToolSearch select:WebSearch,WebFetch) to fill the biggest gaps.',
  '', 'CONTEXT — ' + STEPPE,
  '', 'THE ANGLE REPORTS:\n' + findings.filter(Boolean).map((f,i)=>'### '+ANGLES[i].key+'\n'+f).join('\n\n'),
  '', 'Return: the gaps you found + your additional sourced desiderata (same structured shape), so the synthesis is comprehensive.',
].join('\n'), { label: 'synthesize:critic', phase: 'Synthesize' })

const report = await agent([
  'You are the lead synthesizer. Produce a PRIORITIZED, CITED feature-desirability report for steppe from the angle reports + the critic gaps. Map EACH desired feature to steppe: ALREADY HAVE / PARTIAL / LACKS, and rank by (demand strength) x (fit with steppes GPU precompute-once-fit-many + batched-rotation strength) x (effort). Be honest about demand evidence (anecdotal vs widely-cited). No fabrication.',
  '', 'CONTEXT — ' + STEPPE,
  '', 'ANGLE REPORTS:\n' + findings.filter(Boolean).map((f,i)=>'### '+ANGLES[i].key+'\n'+f).join('\n\n'),
  '', 'CRITIC GAPS:\n' + (critic || '(none)'),
  '', 'Return a clean markdown report: (1) Executive summary — the top 5-8 highest-value features for steppe to build next, each with one-line rationale + demand evidence; (2) a full table [Feature | What/why | Demand evidence (cited) | steppe status (have/partial/lacks) | GPU-fit | rough effort]; (3) a short "quick wins" vs "big bets" split; (4) sources list (real URLs). This report is the deliverable — make it genuinely useful for deciding steppes roadmap.',
].join('\n'), { label: 'synthesize:report', phase: 'Synthesize' })

log('research-desirable-features: synthesis complete')
return { report, critic, findings }
