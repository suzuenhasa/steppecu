# golden_fit1_generate.R — pin an nr>32 AT2 qpAdm golden (the gesvd FALLBACK path
# fixture for M(fit-2)). REAL AADR only (the SAME v66.p1_HO dataset as golden_fit0;
# admixtools 2.0.10, R 4.3.3). Model: the golden_fit0 target+left (nl=2) with a
# LARGE diverse outgroup right set so nr = length(right)-1 > 32 ⇒ the per-model
# gesvd dispatch branch (REPORTED; the executed SVD is the on-device Jacobi). The
# rankdrop/popdrop + §12 metadata are captured into golden_fit1_NNpop.json and the
# f2 tensor dumped to fixtures/f2_fit1_NNpop.bin (the SAME binary layout as
# f2_fit0_9pop.bin: <i P, <i nb, <i*nb block_sizes, <d*(P*P*nb) column-major).
suppressMessages(library(admixtools)); suppressMessages(library(jsonlite))
options(width=220, digits=15)

pref    <- "/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB"
outdir  <- "/workspace/data/aadr/f2_fit1_NRBIG"
target  <- "England_BellBeaker"
left    <- c("Czechia_EBA_CordedWare", "Turkey_N")

# A globally diverse outgroup set (all verified present in the dataset). Kept to a
# size giving nr = length(right)-1 > 32. The set spans Africa / W-Eurasia / E-Asia
# / Oceania / Americas / S-Asia / Siberia so the f4 matrix is well-conditioned.
right_all <- c("Mbuti","Israel_Natufian","Iran_GanjDareh_N","Han","Papuan","Karitiana",
               "YRI","ESN","GWD","LWK","MSL","ACB","Yoruba","Biaka",
               "Mayan","Surui","Pima","Chukchi","Nganasan","Yakut","Ulchi","Dai",
               "Eskimo_Naukan","Aleut","Tibetan","She","Naxi","Miao","Tujia",
               "Kalash","Brahui","Balochi","Makrani","Pathan","Burusho",
               "Mozabite","BantuKenya","Ju_hoan_North","Kusunda","Somali")

# Keep only pops actually present (defensive).
ind <- read.table(paste0(pref,".ind"), stringsAsFactors=FALSE)
present <- unique(ind[[3]])
right <- right_all[right_all %in% present]
stopifnot(all(c(target,left) %in% present))
cat("nr (length(right)-1) =", length(right)-1, "  [>32 required]\n")
stopifnot(length(right)-1 > 32)

allpops <- c(target, left, right)
blgsize <- 0.05; maxmiss <- 0; fudge <- 1e-4

cat("\n--- extract_f2 (REAL AADR; reuse cache if present) ---\n")
if (!dir.exists(outdir) || length(list.files(outdir)) == 0) {
  extract_f2(pref, outdir, pops=allpops, blgsize=blgsize, maxmiss=maxmiss,
             overwrite=TRUE, n_cores=8, verbose=TRUE)
} else {
  cat("  reusing cached f2 at", outdir, "\n")
}

# Read the f2 tensor ONCE; the golden rankdrop is computed FROM THIS SAME f2 object
# so the committed golden and the dumped fixture are BIT-CONSISTENT (avoids the
# directory-path vs read_f2 f2-read-arg ~1e-4 artifact documented in golden_fit0).
f2b <- read_f2(outdir, pops=allpops)          # named [P,P,nb] array

set.seed(42)
res <- qpadm(f2b, left=left, right=right, target=target,
             boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=TRUE)
# Also run the CANONICAL directory-path form for cross-reference (reported only).
set.seed(42)
res_dir <- qpadm(outdir, left=left, right=right, target=target,
                 boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=FALSE)
cat("\n[xref] read_f2 chisq[rank1] =", res$rankdrop$chisq[1],
    " dir-path chisq[rank1] =", res_dir$rankdrop$chisq[1],
    " (golden uses the read_f2 form, fixture-consistent)\n")

cat("\n======== WEIGHTS ========\n");  print(as.data.frame(res$weights))
cat("\n======== RANKDROP ========\n"); print(as.data.frame(res$rankdrop))
cat("\n======== POPDROP ========\n");  print(as.data.frame(res$popdrop))

# ---- the f2 tensor in steppe [P x P x nb] layout, pop order = allpops ----
P   <- length(allpops); nb <- dim(f2b)[3]
# reorder by allpops (read_f2 returns dimnames; enforce the steppe index order)
ord <- match(allpops, dimnames(f2b)[[1]]); stopifnot(!any(is.na(ord)))
f2o <- f2b[ord, ord, , drop=FALSE]
bl  <- attr(f2b, "block_lengths")
if (is.null(bl)) bl <- as.numeric(sub("^l","", dimnames(f2b)[[3]]))
stopifnot(length(bl)==nb)

fix <- "/workspace/steppe/tests/reference/goldens/at2/fixtures/f2_fit1_NRBIG.bin"
con <- file(fix, "wb")
writeBin(as.integer(P),  con, size=4, endian="little")
writeBin(as.integer(nb), con, size=4, endian="little")
writeBin(as.integer(round(bl)), con, size=4, endian="little")
# column-major i + P*j + P*P*b: aperm so the fastest axis is i (rows), then j, then b.
writeBin(as.double(as.vector(f2o)), con, size=8, endian="little")  # R array is already col-major (i,j,b)
close(con)
cat("\nwrote fixture:", fix, " P=",P," nb=",nb," sum(bl)=",sum(round(bl)),"\n")

# ---- §12 metadata + the rankdrop/popdrop JSON (same schema as golden_fit0) ----
sha <- function(p) tryCatch(unname(tools::sha256sum(p)), error=function(e) NA_character_)
rd <- as.data.frame(res$rankdrop); pd <- as.data.frame(res$popdrop)
na2 <- function(x) ifelse(is.na(x), "NA", x)
golden <- list(
  schema = "steppe.golden.at2.qpadm/1",
  purpose = paste0("M(fit-2) nr>32 gesvd-FALLBACK acceptance oracle: the rank test ",
                   "(rankdrop/popdrop) on a >32-right qpAdm model, REAL AADR. ",
                   "Validates the SWEEP MATH at a large model; the executed SVD is the ",
                   "on-device Jacobi (the cuSOLVER gesvd routing is the PENDING seam)."),
  metadata = list(
    R = R.version.string, admixtools = as.character(packageVersion("admixtools")),
    target = target, left = left, right = right,
    nr = length(right)-1, nl = length(left),
    blgsize = blgsize, maxmiss = maxmiss, fudge = fudge, boot = FALSE, seed = 42,
    allsnps = (maxmiss==0), dataset = "v66.p1_HO.aadr.patch.PUB",
    geno_sha256 = sha(paste0(pref,".geno")),
    snp_sha256  = sha(paste0(pref,".snp")),
    ind_sha256  = sha(paste0(pref,".ind"))),
  weights = list(target = target, left = left,
                 weight = res$weights$weight, se = res$weights$se, z = res$weights$z),
  rankdrop = list(f4rank = rd$f4rank, dof = rd$dof, chisq = rd$chisq, p = rd$p,
                  dofdiff = na2(rd$dofdiff), chisqdiff = na2(rd$chisqdiff),
                  p_nested = na2(rd$p_nested)),
  popdrop = list(pat = pd$pat, wt = pd$wt, dof = pd$dof, chisq = pd$chisq, p = pd$p,
                 f4rank = pd$f4rank, feasible = pd$feasible),
  fixture = list(file = "fixtures/f2_fit1_NRBIG.bin", P = P, n_block = nb,
                 layout = "int32 P; int32 nb; int32[nb] block_sizes; float64[P*P*nb] col-major i+P*j+P*P*b",
                 pop_order = allpops)
)
js <- toJSON(golden, pretty=TRUE, auto_unbox=TRUE, digits=15, na="string")
out <- "/workspace/steppe/tests/reference/goldens/at2/golden_fit1_NRBIG.json"
writeLines(js, out)
cat("wrote golden:", out, "\n")
cat("\nDONE. nr=", length(right)-1, " f4rank=", res$rankdrop$f4rank[1], "\n")
