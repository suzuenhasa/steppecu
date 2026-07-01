# /workspace/data/aadr/aadr/01_download.sh
#!/usr/bin/env bash
# aadr/01_download.sh — download the recommended AADR Human Origins (HO) triple
# from Harvard Dataverse via the file-access API.
#
# Source: AADR (Allen Ancient DNA Resource), Harvard Dataverse  DOI: 10.7910/DVN/FFIDCW
# Recommended set (research): HO / Human Origins  ->  *_HO.aadr.patch.PUB.{geno,snp,ind}
#   - the present-day Affymetrix Human Origins moderns are dense and contain many
#     closely-related modern pops => maximal small-f2 catastrophic-cancellation stress.
#
# Dataverse file-access URL form (303-redirects to signed S3; wget -L/--continue follows):
#   https://dataverse.harvard.edu/api/access/datafile/<fileId>
#
# Idempotent: a file already present with the EXPECTED byte size (if known) is skipped;
# if EXPECTED is empty/unverified we accept any non-empty download but warn.
#
# FIXME (UNVERIFIED CONSTANTS): the fileIds, expected byte sizes, the exact version
#   string, and the filenames below were NOT verified against a live Dataverse query in
#   this environment (no network access at authoring time). Before running, confirm the
#   current AADR release and its per-file ids/sizes, e.g.:
#       curl -s "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/FFIDCW" \
#         | python -c 'import json,sys; d=json.load(sys.stdin)["data"]["latestVersion"]; \
#           [print(f["dataFile"]["id"], f["dataFile"]["filesize"], f["dataFile"]["filename"]) for f in d["files"]]'
#   then paste the real id/size/name for the .geno/.snp/.ind of the HO patch.PUB set.
#   Leaving SZ_* empty disables the strict size check (a partial/HTML-error download
#   would then slip through), so prefer to fill in the verified sizes.
set -euo pipefail

echo "==> [01_download] starting"

WORK=/workspace/data/aadr
RAW="${WORK}/raw"
mkdir -p "${RAW}"

command -v wget >/dev/null 2>&1 || { echo "ERROR: wget not found on PATH" >&2; exit 1; }
command -v stat >/dev/null 2>&1 || { echo "ERROR: stat not found on PATH" >&2; exit 1; }

API="https://dataverse.harvard.edu/api/access/datafile"

# -------- file IDs (parameterized) — FILL IN VERIFIED VALUES (see FIXME above) -----
# fileId  expected_bytes (empty = unknown, skip strict check)  local_filename
# VERIFIED against Dataverse API for AADR v66.p1 (DOI 10.7910/DVN/FFIDCW, dataset version 14.0, 2026-06-08).
FID_GENO="13994808" ; SZ_GENO="4029634650" ; FN_GENO="v66.p1_HO.aadr.patch.PUB.geno"
FID_SNP="13994527"  ; SZ_SNP="36800253"    ; FN_SNP="v66.p1_HO.aadr.patch.PUB.snp"
FID_IND="13994526"  ; SZ_IND="1176253"     ; FN_IND="v66.p1_HO.aadr.patch.PUB.ind"
# Optional metadata .anno (not needed for the matrix; uncomment + verify to fetch):
# FID_ANNO="" ; SZ_ANNO="" ; FN_ANNO="v66.p1_HO.aadr.PUB.anno"

# ---------------------------------------------------------------------------
# fetch <fileId> <expected_bytes_or_empty> <dest_filename>
fetch() {
  local fid="$1" want="$2" name="$3"
  local dest="${RAW}/${name}"
  if [[ -z "${fid}" ]]; then
    echo "ERROR: fileId for ${name} is empty — fill in the verified Dataverse id (see FIXME)" >&2
    exit 1
  fi
  if [[ -f "${dest}" ]]; then
    local have; have=$(stat -c %s "${dest}")
    if [[ -n "${want}" && "${have}" == "${want}" ]]; then
      echo "==> [01_download] SKIP ${name} (present, ${have} bytes == expected)"
      return 0
    fi
    if [[ -z "${want}" && "${have}" -gt 0 ]]; then
      echo "==> [01_download] SKIP ${name} (present, ${have} bytes; expected size unverified)"
      return 0
    fi
    echo "==> [01_download] ${name} present but ${have} bytes != expected ${want}; re-downloading"
    rm -f "${dest}"
  fi
  echo "==> [01_download] GET ${name}  (fileId=${fid}, expecting ${want:-<unknown>} bytes)"
  # --continue resumes a partial transfer; -O fixes the output name; -L follows the 303.
  wget --continue -L --tries=5 --timeout=60 -O "${dest}" "${API}/${fid}"
  local got; got=$(stat -c %s "${dest}")
  if [[ "${got}" -le 0 ]]; then
    echo "ERROR: ${name} downloaded 0 bytes" >&2; exit 1
  fi
  if [[ -n "${want}" && "${got}" != "${want}" ]]; then
    echo "ERROR: ${name} size mismatch: got ${got}, expected ${want}" >&2; exit 1
  fi
  if [[ -z "${want}" ]]; then
    echo "==> [01_download] WARNING: ${name} size unverified (got ${got} bytes); set SZ_* to enforce"
  fi
  echo "==> [01_download] OK ${name} (${got} bytes)"
}

fetch "${FID_IND}"  "${SZ_IND}"  "${FN_IND}"
fetch "${FID_SNP}"  "${SZ_SNP}"  "${FN_SNP}"
fetch "${FID_GENO}" "${SZ_GENO}" "${FN_GENO}"
# fetch "${FID_ANNO}" "${SZ_ANNO}" "${FN_ANNO}"

# Cheap format sanity: the .geno must start with the EIGENSOFT 'GENO' magic.
if ! head -c 4 "${RAW}/${FN_GENO}" | grep -q '^GENO'; then
  echo "ERROR: ${FN_GENO} does not start with 'GENO' magic — likely an HTML/error page, not a packed .geno" >&2
  exit 1
fi

# Stable symlinks so 02_build_matrix.py has a fixed input prefix.
PREFIX="${RAW}/HO"
ln -sf "${FN_GENO}" "${PREFIX}.geno"
ln -sf "${FN_SNP}"  "${PREFIX}.snp"
ln -sf "${FN_IND}"  "${PREFIX}.ind"
echo "==> [01_download] stable prefix: ${PREFIX}.{geno,snp,ind}"

echo "==> [01_download] SIZES"
echo "----------------------------------------"
ls -l "${RAW}/${FN_GENO}" "${RAW}/${FN_SNP}" "${RAW}/${FN_IND}"
du -h "${RAW}/${FN_GENO}" "${RAW}/${FN_SNP}" "${RAW}/${FN_IND}"
echo "----------------------------------------"

echo "==> [01_download] DONE"
