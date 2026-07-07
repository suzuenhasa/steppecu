#!/usr/bin/env python3
"""Decode 38 kept AADR samples from the v66 1240K TGENO directly into a PLINK
bed/bim/fam subset (autosomes only), using the validated TGENO layout from
build_tgeno_matrix.py. Independently validates the decode against the .anno
per-sample coverage (col27) before writing."""
import csv, math, os, sys
import numpy as np

D="/workspace/data/aadr/1240k"
W="/workspace/readv2_oracle"
GENO=f"{D}/v66.p1_1240K.aadr.patch.PUB.geno"
SNP =f"{D}/v66.p1_1240K.aadr.patch.PUB.snp"
IND =f"{D}/v66.p1_1240K.aadr.patch.PUB.ind"
ANNO=f"{D}/v66.p1_1240K.aadr.PUB.anno"
OUT =f"{W}/readv2_subset"

keep=[l.strip() for l in open(f"{W}/keep_genetic_ids.txt") if l.strip()]
keepset=set(keep)

# ---- TGENO header ----
with open(GENO,"rb") as f: head=f.read(48)
parts=head.split(b"\x00")[0].split()
assert parts[0]==b"TGENO", parts[:1]
n_ind,n_snp=int(parts[1]),int(parts[2])
bpi=math.ceil(n_snp/4)
fsize=os.path.getsize(GENO)
hdr=fsize-n_ind*bpi
print(f"[hdr] n_ind={n_ind} n_snp={n_snp} bytes/ind={bpi} header={hdr}", file=sys.stderr)

# ---- .ind: row index per genetic ID + group ----
ind_rows={}; ind_group={}
with open(IND) as f:
    for i,line in enumerate(f):
        t=line.split()
        if len(t)>=3:
            ind_rows[t[0]]=i; ind_group[t[0]]=t[2]
for g in keep: assert g in ind_rows, f"{g} not in .ind"

# ---- .snp: parse; autosomal mask (chr 1..22) ----
chrom=[]; sid=[]; cm=[]; bp=[]; a1=[]; a2=[]
with open(SNP) as f:
    for line in f:
        t=line.split()
        # EIGENSTRAT .snp: id chr gpos(Morgans) physpos allele1(REF) allele2(ALT)
        sid.append(t[0]); chrom.append(t[1]); cm.append(t[2]); bp.append(t[3])
        a1.append(t[4]); a2.append(t[5])
chrom=np.array(chrom); sid=np.array(sid); cm=np.array(cm); bp=np.array(bp)
a1=np.array(a1); a2=np.array(a2)
assert len(sid)==n_snp, (len(sid),n_snp)
def is_auto(c):
    try: return 1<=int(c)<=22
    except: return False
auto=np.array([is_auto(c) for c in chrom])
print(f"[snp] total={n_snp} autosomal={auto.sum()}", file=sys.stderr)
auto_idx=np.nonzero(auto)[0]

# ---- decode each kept sample over autosomal SNPs ----
mm=np.memmap(GENO,dtype=np.uint8,mode="r")
def decode(row):
    off=hdr+row*bpi
    buf=mm[off:off+bpi]
    bits=np.unpackbits(buf)                 # MSB-first
    return (bits[0::2]*2+bits[1::2])[:n_snp].astype(np.int8)

C=np.zeros((len(keep),auto_idx.size),dtype=np.int8)  # samples x autosomal SNPs, {0,1,2,3}
ploidy={}; nonmiss={}
NTEST=1000
for k,g in enumerate(keep):
    full=decode(ind_rows[g])
    det=full[:min(NTEST,n_snp)]
    ploidy[g]=2 if np.any(det==1) else 1
    ca=full[auto_idx]
    C[k]=ca
    nonmiss[g]=int(np.sum(ca!=3))

# ---- VALIDATION 1: all pseudo-haploid ----
dips=[g for g in keep if ploidy[g]!=1]
print(f"[ploidy] pseudo-haploid={sum(1 for g in keep if ploidy[g]==1)} diploid={len(dips)}", file=sys.stderr)
assert not dips, f"DIPLOID samples present (v1 rejects): {dips}"

# also assert no het codes anywhere in the autosomal matrix (pseudo-haploid invariant)
n_het=int(np.sum(C==1))
print(f"[het] het codes in autosomal matrix (must be 0 for PH): {n_het}", file=sys.stderr)
assert n_het==0, "het calls present in supposedly pseudo-haploid samples"

# ---- VALIDATION 2: per-sample nonmissing vs anno col27 ----
anno_snps={}
with open(ANNO,newline="") as f:
    r=csv.reader(f,delimiter="\t"); next(r)
    for row in r:
        if len(row)>=32:
            g=row[0].strip()
            if g in keepset:
                v=row[26].strip().replace(",","")
                try: anno_snps[g]=int(float(v))
                except: anno_snps[g]=None
print("[validate] per-sample decoded-nonmissing vs anno col27 (SNPs hit 1240k):", file=sys.stderr)
maxrel=0.0
for g in keep:
    a=anno_snps.get(g); d=nonmiss[g]
    rel=abs(d-a)/a if a else float('nan')
    maxrel=max(maxrel,rel if a else 0)
    flag="" if (a and rel<0.02) else "  <-- CHECK"
    print(f"    {g:<14} decoded={d:>8}  anno={a}  reldiff={rel:.4f}{flag}", file=sys.stderr)
print(f"[validate] max reldiff = {maxrel:.4f}", file=sys.stderr)

# ---- write .fam ----
with open(OUT+".fam","w") as f:
    for g in keep:
        f.write(f"{ind_group[g]}\t{g}\t0\t0\t0\t-9\n")

# ---- write .bim (autosomal): chr id cM bp A1(REF) A2(ALT) ----
with open(OUT+".bim","w") as f:
    for j in auto_idx:
        f.write(f"{chrom[j]}\t{sid[j]}\t{cm[j]}\t{bp[j]}\t{a1[j]}\t{a2[j]}\n")

# ---- write .bed (SNP-major) ----
# bed 2-bit: 00=homA1, 01=missing, 10=het, 11=homA2. A1=REF => REF-copy code c maps:
#   c=2 (hom REF)  -> 00 ; c=1 (het) -> 10 ; c=0 (hom ALT) -> 11 ; c=3 (missing) -> 01
lut=np.array([0b11, 0b10, 0b00, 0b01], dtype=np.uint8)  # index by c in {0,1,2,3}
Bt=lut[C.T]                                             # (nsnp, nsamp)
nsnp_a,nsamp=Bt.shape
pad=(-nsamp)%4
if pad: Bt=np.concatenate([Bt,np.zeros((nsnp_a,pad),dtype=np.uint8)],axis=1)
Bt=Bt.reshape(nsnp_a,-1,4)
packed=(Bt[:,:,0]|(Bt[:,:,1]<<2)|(Bt[:,:,2]<<4)|(Bt[:,:,3]<<6)).astype(np.uint8)
with open(OUT+".bed","wb") as f:
    f.write(bytes([0x6c,0x1b,0x01]))
    f.write(packed.tobytes())

print(f"[write] {OUT}.bed/.bim/.fam  samples={nsamp} snps={nsnp_a}", file=sys.stderr)
print("OK")
