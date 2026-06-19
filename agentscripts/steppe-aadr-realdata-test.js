export const meta = {
  name: 'steppe-aadr-realdata-test',
  description: 'Real AADR data: build per-pop Q/V/N from TGENO, then run f2 emulated-vs-native accuracy + big-P all-pairs throughput on the 5090 — STRICTLY SERIAL on the GPU (no overlap).',
  phases: [
    { title: 'Prep', detail: 'CPU: build accuracy (P~50) + throughput (P~1151) matrices from TGENO geno' },
    { title: 'Accuracy', detail: 'GPU alone: f2_emu_spike --load on closely-related pops' },
    { title: 'Throughput', detail: 'GPU alone: f2_timing --load on big-P all-pairs matrix' },
    { title: 'Report', detail: 'synthesize real-data vs synthetic; doc-update text' },
  ],
}

const SSH = 'ssh -i ~/.ssh/id_vastai -p 63709 -o ConnectTimeout=25 root@108.255.76.60'
const PY = '/venv/main/bin/python'
const RAW = '/workspace/data/aadr/raw'
const BIN = '/workspace/steppe/experiments/f2_emu_spike'

phase('Prep')
const prep = await agent([
  'SSH to the GPU box and build TWO per-population matrices from the AADR TGENO geno using build_tgeno_matrix.py. This is CPU-only (numpy); run it, do NOT run any GPU work.',
  'Run these two commands (verbatim) and capture the [hdr]/[sel]/[sanity]/[done] stderr lines from each:',
  '',
  '  ' + SSH + " 'cd " + RAW + ' && ' + PY + ' build_tgeno_matrix.py --geno v66.p1_HO.aadr.patch.PUB.geno --ind v66.p1_HO.aadr.patch.PUB.ind --out /workspace/data/aadr/derived_acc --auto-top 50 --snp-cap 100000' + "'",
  '  ' + SSH + " 'cd " + RAW + ' && ' + PY + ' build_tgeno_matrix.py --geno v66.p1_HO.aadr.patch.PUB.geno --ind v66.p1_HO.aadr.patch.PUB.ind --out /workspace/data/aadr/derived_big --min-n 5 --snp-cap 30000' + "'",
  '',
  'For EACH matrix report: P (pops), M (SNPs), mean ref-freq, pair-missing-frac, and whether the sanity check passed (no out-of-range Q). If a build errors or sanity fails (out-of-range Q ⇒ TGENO bit-order bug), report the failure loudly and stop. Return the two matrices summary.',
].join('\n'), { label: 'prep-matrices', phase: 'Prep' })

phase('Accuracy')
const accuracy = await agent([
  'SSH to the GPU box and run the f2 emulated-FP64 vs native-FP64 ACCURACY test on the real closely-related-population matrix. This is GPU work and MUST run alone (no other GPU job concurrent — it is the only thing the workflow runs right now).',
  'Command: ' + SSH + " '" + BIN + '/f2_emu_spike --load /workspace/data/aadr/derived_acc' + "'",
  'Return the FULL output verbatim (the [mem] line, the table header + data row(s), and the verdict line). Then state plainly: maxRel_emu vs maxRel_nat (is emulation as accurate as native?), the mBits value, emuEqNat (must be "no" = emulation engaged), and the verdict (PASS vs FAIL_IGNORED). If FAIL_IGNORED, say emulation did not engage.',
].join('\n'), { label: 'accuracy-run', phase: 'Accuracy' })

phase('Throughput')
const throughput = await agent([
  'SSH to the GPU box and run the f2 native-vs-emulated THROUGHPUT test on the big-P (P~1151) all-pairs real-data matrix. GPU work — runs alone (the accuracy run already finished).',
  'Command: ' + SSH + " '" + BIN + '/f2_timing --load /workspace/data/aadr/derived_big 20' + "'",
  'Return the table row verbatim (P, M, AI~P/8, t_nat_ms, t_emu_ms, emu/nat, emuEqNat, GFLOP). Then state: the emu/nat speedup (>1 = emulation faster), confirm emuEqNat is "no" (emulation engaged), and how this real-data P~1151 result compares to the synthetic crossover (emulation won ~8x at P=2048).',
].join('\n'), { label: 'throughput-run', phase: 'Throughput' })

phase('Report')
const report = await agent([
  'Synthesize the REAL AADR-data results with the prior SYNTHETIC findings into a tight, factual summary plus the precise architecture-doc edit.',
  '',
  'Prior synthetic findings (measured on this 5090): emulated FP64 = native-FP64 accuracy (dynamic mantissa picked 54 bits, matched/beat native through condition number ~3e7); throughput crossover at P~256; emulation ~8x faster at P=2048 (native 2012 ms vs emu 252 ms). Conclusion drawn: Ozaki emulated-FP64 belongs in the big-P all-pairs f2 PRECOMPUTE (compute-bound regime where consumer-5090 native FP64 is ~1/64-rate crippled); native FP64 stays for the small per-model fits and the cancellation-sensitive elementwise/f2-reduction math.',
  '',
  'REAL AADR ACCURACY RESULT:',
  accuracy,
  '',
  'REAL AADR THROUGHPUT RESULT:',
  throughput,
  '',
  'PREP/MATRICES:',
  prep,
  '',
  'Produce: (1) a crisp findings summary stating whether the REAL AADR data CONFIRMS the synthetic conclusion (accuracy parity + big-P emulation speedup), with the actual numbers; (2) the exact text to record in architecture.md §12/§5 — that emulation is the default for the large-P all-pairs precompute and native FP64 for small fits, now backed by both synthetic and real-AADR measurement. Be factual; flag any surprise (e.g. if real-data accuracy or speedup differed from synthetic).',
].join('\n'), { label: 'synthesize-report', phase: 'Report' })

return { prep, accuracy, throughput, report }
