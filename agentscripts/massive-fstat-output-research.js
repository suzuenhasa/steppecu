export const meta = {
  name: 'massive-fstat-output-research',
  description: 'RESEARCH ONLY (no code changes, no implementation): how should steppe handle MASSIVE f-stat output — billions of items (all-quartets/triples over 1000+ pops), a [N×4] est/se/z/p table reaching multi-TB that cannot be RAM-resident? Context: the COMPUTE is already solved (diagonal jackknife + tiling stream in bounded VRAM at any N); the OPEN problem is purely the OUTPUT table. We have NEVER researched this. Explore the full option space — NOT just the streaming-emitter I proposed: (a) PRIOR ART — how ADMIXTOOLS 2 / classic DReichLab ADMIXTOOLS / ALDER / qpDstat + other pop-gen tools handle large batch f-stat/D-stat output (full table vs filtered? what format? on the box /workspace/AdmixTools_src + web); (b) COMPRESSION + COLUMNAR FORMATS — does compression help an [N×4] float table (est/se/z/p)? Parquet/Arrow + Zstd/float codecs, expected ratios for f-stat-shaped numeric data, queryability, mmap, the Python read-back (pyarrow/pandas chunked); (c) BIG-DATA TABULAR-OUTPUT PATTERNS generally — streaming writers, sharded columnar, memory-mapped output, how billions-of-rows numeric tables are produced + consumed at scale; (d) THE SCIENCE/USAGE + FILTERING ANGLE — do users actually WANT all billions of f-stats, or filtered/thresholded (significant-only / |z|>threshold / top-K)? What is the realistic massive use-case + the POST-FILTER output size? (this could drastically change the answer — maybe you never write the full table); (e) STEPPE REUSE — what steppe ALREADY has (the f2 on-disk format f2_disk_format.hpp, the StagingRing/HostRam/Disk sinks block_sink.cuh, the M5 tiering, the OutputFormat emitter result_emit.cpp, the nanobind/numpy bindings seam) that a massive-output design should reuse. Then SYNTHESIZE a recommended output architecture (format + streaming/compression + filtering policy + Python read-back + steppe reuse) with VERIFIED trade-offs. HARD VERIFY MANDATE: cite every claim (web URL / AT2 source file:line / steppe code file:line); FLAG anything speculative; do NOT assert a tool behaves a way without checking. Write docs/research/massive-fstat-output.md. NO implementation, NO code changes (research + the doc only).',
  phases: [ { title: 'Parallel research: prior-art, compression, patterns, usage, steppe-reuse' }, { title: 'Synthesize the recommended output architecture' } ],
}

const R = '/home/suzunik/steppe'
const SSH = 'ssh box5090'
const AT2SRC = '/workspace/AdmixTools_src'

const STD = [
  'PROJECT: steppe = GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm (f4/f3/f4-ratio/qpDstat standalone stats just landed). Branch phase2-fit-engine @ 8751759.',
  'THE PROBLEM (research it, do NOT implement): the standalone f-stats can now COMPUTE at massive scale (the diagonal-jackknife + item-tiling stream in bounded VRAM at any N — verified, OOM-free to 1M items, linear). The UNSOLVED part is the OUTPUT: all-quartets/triples over 1000+ pops = BILLIONS of items, and the result is a [N×4] table (est/se/z/p, 8 bytes each = 32·N bytes) — at billions of rows that is MULTI-TB and cannot be RAM-resident (true in C++ AND Python; a billions-row numpy/pandas object is impossible). We have NEVER researched how to handle this. The user asked specifically: does COMPRESSION help, and how is this handled in general / regardless.',
  'EXPLORE THE FULL OPTION SPACE — do NOT assume the streaming-emitter is the answer. Options to weigh: stream-to-file (CSV/TSV) vs columnar (Parquet/Arrow) + compression (Zstd / float codecs / bit-packing) vs sharded files vs memory-mapped output vs FILTERING (only write significant / |z|>t / top-K — maybe the full table is never wanted) vs a queryable on-disk store. Cover the Python read-back story for each (you cannot return a billions-row DataFrame).',
  'HARD VERIFY MANDATE (the user is emphatic): VERIFY every load-bearing claim against a real source — cite web URLs, the DReichLab AdmixTools C source on the box (' + AT2SRC + ', reachable via ' + SSH + '), the admixtools R package, or steppe code (file:line). FLAG anything you could not verify as speculative. Do NOT state "tool X does Y / format Z compresses W" without checking. The point of the research is grounded fact, not plausible-sounding guesses.',
  'NO IMPLEMENTATION, NO CODE CHANGES. This workflow produces ONLY docs/research/massive-fstat-output.md (the research + the recommendation). Single deliverable doc; no edits to src/.',
].join('\n')

async function tryAgent(p, opts) { let r = await agent(p, opts); if (r === null) { log(opts.label + ': transient null — retry'); r = await agent(p, { ...opts, label: opts.label + ':retry' }) } return r }

const FIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['angle','findings','verified','speculative','sources','recommendation_input'],
  properties: {
    angle: { type: 'string' },
    findings: { type: 'string', description: 'the substantive findings for this angle' },
    verified: { type: 'string', description: 'what was VERIFIED against a real source (with the citation)' },
    speculative: { type: 'string', description: 'what is plausible but NOT verified (flagged honestly)' },
    sources: { type: 'string', description: 'the URLs / file:line / source paths consulted' },
    recommendation_input: { type: 'string', description: 'what this angle contributes to the final output-architecture recommendation' },
  },
}

phase('Parallel research: prior-art, compression, patterns, usage, steppe-reuse')
const angles = [
  { key: 'prior-art', p: 'ANGLE A — PRIOR ART. How do existing pop-gen tools handle LARGE batch f-stat/D-stat output? VERIFY: the DReichLab AdmixTools C (' + SSH + ' ' + AT2SRC + ' — read qpDstat.c / qpAdm / the output writers: do they write a full table or filter? what format? streaming or buffered?); the admixtools 2 R package (qpdstat/f4 return value — a tibble; how does it behave at large N? web github uqrmaie1/admixtools); ALDER / classic ADMIXTOOLS / Dsuite / other ABBA-BABA-at-scale tools (web — how they emit millions+ of D-stats). Do any produce billions of rows, and how (file format, streaming, filtering)? Cite source file:line / URLs.' },
  { key: 'compression-formats', p: 'ANGLE B — COMPRESSION + COLUMNAR FORMATS. Does compression meaningfully help a [N×4] float64 table (est, se, z, p) of f-stats? VERIFY: columnar formats (Apache Parquet, Arrow/Feather) + their compression codecs (Zstd, Snappy, dictionary) and float-specific encodings (byte-stream-split, bit-packing); expected compression RATIOS for f-stat-shaped numeric data (est/se small-magnitude doubles, z, p in [0,1] — what is realistically compressible vs float entropy limits); queryability (predicate pushdown to read back only significant rows) + memory-mapping; the Python read-back (pyarrow.dataset / pandas chunked / polars). Cite the format specs + any benchmark URLs; FLAG ratio estimates that are not from a cited measurement.' },
  { key: 'bigdata-patterns', p: 'ANGLE C — BIG-DATA TABULAR-OUTPUT PATTERNS (general). How is a billions-of-rows numeric table PRODUCED and CONSUMED at scale, regardless of domain? VERIFY/survey: streaming/chunked writers that never hold full-N resident; sharded/partitioned output (many files); memory-mapped arrays (numpy memmap, zarr, HDF5); on-disk queryable stores; the producer-bounded-memory + consumer-chunked-read contract. What is the standard, robust pattern? Cite library docs/URLs (zarr, HDF5, parquet, numpy.memmap). What are the practical limits (disk capacity for multi-TB, write throughput)?' },
  { key: 'usage-filtering', p: 'ANGLE D — SCIENCE/USAGE + FILTERING (possibly the most important). Do researchers actually WANT all billions of f-stats from an all-quartets sweep, or do they FILTER (significant only, |z|>3, top-K, a specific hypothesis set)? VERIFY against real pop-gen practice (web — how f4/D-stat scans are used in papers; do people enumerate ALL quartets or a targeted set?; what is the realistic massive use-case). If the full billions-table is rarely wanted, the right design may be ON-THE-FLY FILTERING (write only rows passing a threshold) — which could shrink multi-TB to GBs and change everything. Quantify: for a realistic |z|>threshold, what fraction of billions survives? Cite sources; FLAG where you are reasoning vs citing.' },
  { key: 'steppe-reuse', p: 'ANGLE E — STEPPE REUSE. What does steppe ALREADY have that a massive-output design should reuse? READ (file:line): src/device/f2_disk_format.hpp (the f2 on-disk format — could the f-stat output mirror it?), src/device/cuda/block_sink.cuh (the StagingRing triple-buffered pinned ring + HostRam/Disk sinks + the background writer + DrainFn), the M5 tiering (tier_select.hpp), src/app/result_emit.cpp (the current OutputFormat CSV/TSV/JSON emitter — how it writes today), the f-stat run_* result types (f4.cpp/dstat.cpp — the [N×4] vectors), the nanobind bindings (bindings/module.cpp — the numpy/DataFrame return, the DLPack/__cuda_array_interface__ seam). What is directly reusable for a streamed/columnar/compressed f-stat output? Cite file:line.' },
]
const research = await parallel(angles.map(a => () => tryAgent([
  'You are a research agent. ' + a.p, STD,
].join('\n'), { schema: FIND_SCHEMA, label: 'research:' + a.key, phase: 'Parallel research: prior-art, compression, patterns, usage, steppe-reuse' })))
const ok = research.filter(Boolean)
log('research angles returned: ' + ok.length + '/' + angles.length)
if (ok.length === 0) { log('--- all research died — HALT'); return { halted: true } }

phase('Synthesize the recommended output architecture')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['recommendation','compression_verdict','filtering_verdict','format_choice','python_readback','steppe_reuse','open_decisions','doc_committed','note'],
  properties: {
    recommendation: { type: 'string', description: 'the recommended massive-f-stat-output architecture (the headline answer)' },
    compression_verdict: { type: 'string', description: 'does compression help, and how much (the user asked specifically) — verified ratios if available' },
    filtering_verdict: { type: 'string', description: 'should we filter (write only significant) rather than the full table — and does that change everything' },
    format_choice: { type: 'string', description: 'CSV vs Parquet/Arrow vs sharded vs mmap/zarr/HDF5 — the recommended format + why' },
    python_readback: { type: 'string', description: 'how Python consumes massive output (cannot be a billions-row DataFrame)' },
    steppe_reuse: { type: 'string', description: 'what existing steppe machinery the design reuses (file:line)' },
    open_decisions: { type: 'string', description: 'what still needs a user decision before implementing' },
    doc_committed: { type: 'string', description: 'the commit hash of docs/research/massive-fstat-output.md' },
    note: { type: 'string', description: 'how confident / what is verified vs speculative' },
  },
}
const synth = await tryAgent([
  'You are synthesizing the massive-f-stat-output research into a recommended architecture. The angle findings:\n<<<\n' + JSON.stringify(ok) + '\n>>>', STD, '',
  'WRITE docs/research/massive-fstat-output.md: the problem framing; per-angle findings (prior art, compression, big-data patterns, usage/filtering, steppe reuse) with citations; the OPTION COMPARISON (stream-to-file vs columnar+compression vs sharded vs filtered vs mmap/store) and their trade-offs; the RECOMMENDED architecture (format + streaming/compression + filtering policy + Python read-back + steppe reuse); a clear VERIFIED-vs-SPECULATIVE split; and the OPEN DECISIONS the user must make before any implementation. Be honest where the research is thin. Then cd ' + R + ' && git add ONLY docs/research/massive-fstat-output.md, commit (research(massive-fstat-output): how to handle billions-of-items f-stat output — compression / columnar / filtering / streaming options + recommendation) ending: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com> . Capture the hash. Return the structured synthesis.',
].join('\n'), { schema: SYNTH_SCHEMA, label: 'synth:output', phase: 'Synthesize the recommended output architecture' })
if (synth === null) { log('--- synth died — HALT'); return { halted: true, research: ok } }
log('PERF-OUTPUT research: ' + String(synth.recommendation).slice(0,160))
return { research: ok, synth }
