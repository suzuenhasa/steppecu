# golden_rot_generate.R — pin the REAL AT2 model-space ROTATION golden for M(fit-6)
# S8 (the qpAdm rotation/search acceptance oracle). REAL AADR only (the SAME
# v66.p1_HO dataset as golden_fit0/golden_fit1; admixtools 2.0.10, R 4.3.3,
# boot=FALSE deterministic, set.seed(42)).
#
# The rotation: ONE target + ONE fixed right set + a POOL of candidate left
# sources. ENUMERATE every 2-source AND 3-source subset of the pool (the AT2
# "rotation" idiom: qpadm per model, the subset as `left`, the fixed `right`,
# the fixed `target`). For EACH model capture est (weights), se, z, p (tail of
# f4rank), f4rank, and the feasibility flag — the per-model ranked feasibility
# table the GPU batched rotation (run_qpadm_search) is diffed against.
#
# The right set is the golden_fit0 6-pop outgroup set (nr=5 <= 32) so the
# VALIDATED set exercises the gesvdjBatched common path (the S8 design center).
# Every pool source is verified ABSENT from the right set (no left/right overlap).
#
# The f2 tensor over the union pop set is dumped to fixtures/f2_rot.bin in the
# EXISTING binary layout (<i P, <i nb, <i*nb block_sizes, <d*(P*P*nb) col-major
# i+P*j+P*P*b), and the golden rows are computed FROM THE SAME read_f2 object so
# the committed golden and the dumped fixture are BIT-CONSISTENT.
suppressMessages(library(admixtools)); suppressMessages(library(jsonlite))
options(width=220, digits=15)

pref    <- "/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB"
outdir  <- "/workspace/data/aadr/f2_rot"
target  <- "England_BellBeaker"

# The candidate SOURCE POOL (8 pops). All verified present in the .ind and all
# DISJOINT from the right set below (the contract: exclude any pool source also
# in right). Pop-gen-meaningful: the true England_BellBeaker pair
# (Czechia_EBA_CordedWare + Turkey_N, the golden_fit0 model) is inside the pool,
# so the rotation surfaces a feasible 2-way model amid infeasible competitors.
pool <- c("Czechia_EBA_CordedWare",          # Corded Ware (steppe-ancestry source)
          "Turkey_N",                        # Anatolian Neolithic farmer
          "Russia_Samara_EBA_Yamnaya",       # Yamnaya steppe
          "Luxembourg_Loschbour_Mesolithic", # WHG
          "Russia_Karelia_Mesolithic",       # EHG
          "Spain_EN",                        # Iberian Early Neolithic farmer
          "England_N",                       # British Neolithic
          "Russia_Khakassia_Afanasievo")     # Afanasievo steppe

# The fixed right / outgroup set (golden_fit0 6-pop; nr = length(right)-1 = 5).
right <- c("Mbuti","Israel_Natufian","Iran_GanjDareh_N","Han","Papuan","Karitiana")

# ---- defensive presence + no-overlap checks --------------------------------
ind <- read.table(paste0(pref,".ind"), stringsAsFactors=FALSE)
present <- unique(ind[[3]])
stopifnot(all(c(target, pool, right) %in% present))
stopifnot(length(intersect(pool, right)) == 0)   # contract: no left/right overlap
stopifnot(!(target %in% pool), !(target %in% right))
cat("target:", target, "\n")
cat("pool (", length(pool), "):", paste(pool, collapse=", "), "\n")
cat("right (nr=", length(right)-1, "):", paste(right, collapse=", "), "\n")

allpops <- c(target, pool, right)
blgsize <- 0.05; maxmiss <- 0; fudge <- 1e-4

cat("\n--- extract_f2 (REAL AADR; reuse cache if present) ---\n")
if (!dir.exists(outdir) || length(list.files(outdir)) == 0) {
  extract_f2(pref, outdir, pops=allpops, blgsize=blgsize, maxmiss=maxmiss,
             overwrite=TRUE, n_cores=8, verbose=TRUE)
} else {
  cat("  reusing cached f2 at", outdir, "\n")
}

# Read the f2 tensor ONCE; the golden rows are computed FROM THIS SAME object so
# the committed golden + the dumped fixture are BIT-CONSISTENT (the golden_fit1
# discipline; avoids the dir-path vs read_f2 ~1e-4 artifact).
f2b <- read_f2(outdir, pops=allpops)          # named [P,P,nb] array

# ---- enumerate the rotation: all 2- and 3-source subsets of the pool --------
subsets <- c(
  utils::combn(pool, 2, simplify=FALSE),
  utils::combn(pool, 3, simplify=FALSE))
cat("\n--- rotation:", length(subsets), "models (",
    choose(length(pool),2), "2-source +", choose(length(pool),3), "3-source) ---\n")

models <- vector("list", length(subsets))
for (i in seq_along(subsets)) {
  left_i <- subsets[[i]]
  set.seed(42)
  res <- tryCatch(
    qpadm(f2b, left=left_i, right=right, target=target,
          boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=FALSE),
    error = function(e) e)

  if (inherits(res, "error")) {
    # A degenerate model (AT2 itself failed) — record it honestly, no fabrication.
    models[[i]] <- list(model_index = i-1L, left = left_i,
                        weight = NA, se = NA, z = NA, p = NA, f4rank = NA,
                        feasible = NA, status = paste0("at2_error: ", conditionMessage(res)))
    cat(sprintf("[%3d] {%s}  AT2 ERROR: %s\n", i-1L,
                paste(left_i, collapse=","), conditionMessage(res)))
    next
  }

  w  <- res$weights
  rd <- as.data.frame(res$rankdrop)
  pd <- as.data.frame(res$popdrop)
  # The fitted-model row is the all-pops-present popdrop pattern (the first row,
  # pat == "0...0"): its p is the model tail-p at f4rank, feasible is its flag.
  fit_row <- pd[1, , drop=FALSE]
  p_fit       <- fit_row$p
  f4rank_fit  <- fit_row$f4rank
  feasible_fit<- as.logical(fit_row$feasible)

  models[[i]] <- list(
    model_index = i-1L,
    left   = left_i,
    weight = as.numeric(w$weight),
    se     = as.numeric(w$se),
    z      = as.numeric(w$z),
    p      = p_fit,
    f4rank = as.integer(f4rank_fit),
    feasible = feasible_fit,
    status = "ok")

  cat(sprintf("[%3d] {%s}  p=%.4g f4rank=%d feasible=%s  w=[%s]\n",
              i-1L, paste(left_i, collapse=","), p_fit, f4rank_fit, feasible_fit,
              paste(sprintf("%.4f", w$weight), collapse=",")))
}

# ---- dump the f2 tensor (steppe [P x P x nb] layout, pop order = allpops) ----
P   <- length(allpops); nb <- dim(f2b)[3]
ord <- match(allpops, dimnames(f2b)[[1]]); stopifnot(!any(is.na(ord)))
f2o <- f2b[ord, ord, , drop=FALSE]
bl  <- attr(f2b, "block_lengths")
if (is.null(bl)) bl <- as.numeric(sub("^l","", dimnames(f2b)[[3]]))
stopifnot(length(bl)==nb)

fix <- "/workspace/steppe/tests/reference/goldens/at2/fixtures/f2_rot.bin"
con <- file(fix, "wb")
writeBin(as.integer(P),  con, size=4, endian="little")
writeBin(as.integer(nb), con, size=4, endian="little")
writeBin(as.integer(round(bl)), con, size=4, endian="little")
writeBin(as.double(as.vector(f2o)), con, size=8, endian="little")  # R array col-major (i,j,b)
close(con)
cat("\nwrote fixture:", fix, " P=",P," nb=",nb," sum(bl)=",sum(round(bl)),"\n")

# ---- §12 metadata + the rotation rows JSON ---------------------------------
# sha256 via the system tool (tools::sha256sum can NA-out on the 4GB geno under
# some R builds; the system `sha256sum` is robust and is the §12 record of truth).
sha <- function(p) {
  out <- tryCatch(system2("sha256sum", shQuote(p), stdout=TRUE, stderr=FALSE),
                  error=function(e) NA_character_)
  if (length(out) == 0 || is.na(out[1])) return(NA_character_)
  sub("\\s.*$", "", out[1])
}
golden <- list(
  schema  = "steppe.golden.at2.qpadm.rotation/1",
  purpose = paste0("M(fit-6) S8 ROTATION acceptance oracle: per-model qpAdm fit + ",
                   "rank test over all 2- and 3-source subsets of an 8-pop source ",
                   "POOL (fixed target + fixed nr<=32 right set), REAL AADR. The GPU ",
                   "BATCHED rotation (run_qpadm_search; gesvdjBatched + batched solves, ",
                   "f2 resident, model-list sharded across both 5090s) is diffed ",
                   "per-model against these rows (weights rtol~1e-6, p/feasibility ",
                   "loose/decision, f4rank exact)."),
  metadata = list(
    R = R.version.string, admixtools = as.character(packageVersion("admixtools")),
    target = target, pool = pool, right = right,
    n_models = length(subsets), n_pool = length(pool),
    subset_sizes = c(2L, 3L),
    nr = length(right)-1,
    blgsize = blgsize, maxmiss = maxmiss, fudge = fudge, boot = FALSE, seed = 42,
    als_iterations = 20, rank_alpha = 0.05,
    allsnps = (maxmiss==0), dataset = "v66.p1_HO.aadr.patch.PUB",
    geno_sha256 = sha(paste0(pref,".geno")),
    snp_sha256  = sha(paste0(pref,".snp")),
    ind_sha256  = sha(paste0(pref,".ind"))),
  models = models,
  fixture = list(file = "fixtures/f2_rot.bin", P = P, n_block = nb,
                 layout = "int32 P; int32 nb; int32[nb] block_sizes; float64[P*P*nb] col-major i+P*j+P*P*b",
                 pop_order = allpops)
)
js <- toJSON(golden, pretty=TRUE, auto_unbox=TRUE, digits=15, na="string")
out <- "/workspace/steppe/tests/reference/goldens/at2/golden_rot.json"
writeLines(js, out)
cat("wrote golden:", out, "\n")

nfeas <- sum(vapply(models, function(m) isTRUE(m$feasible), logical(1)))
cat("\nDONE.", length(subsets), "models;", nfeas, "feasible.\n")
