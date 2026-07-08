#!/usr/bin/env bash
# steppe installer: fetch the prebuilt CLI, install a CUDA-aware launcher, and preflight.
#
#   curl -fsSL https://raw.githubusercontent.com/suzuenhasa/steppecu/main/install.sh | bash
#   bash install.sh --dir ~/.local/bin --tag v0.2.0
#
# steppe needs a CUDA 13 runtime. This diagnoses that up front with an actionable message
# (not the opaque "libcudart.so.13: cannot open shared object file" loader error), and the
# installed `steppe` launcher re-checks + sets LD_LIBRARY_PATH on every run.
set -euo pipefail

REPO="suzuenhasa/steppecu"
TAG="v0.2.0"
DIR="${HOME}/.local/bin"

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) echo "usage: install.sh [--dir DIR] [--tag TAG]"; exit 0 ;;
    *) echo "install.sh: unknown arg: $1" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
err() { printf 'steppe: %s\n' "$*" >&2; }

# --- platform -------------------------------------------------------------------------
os="$(uname -s)"; arch="$(uname -m)"
if [ "$os" != "Linux" ] || [ "$arch" != "x86_64" ]; then
  err "only Linux x86_64 is supported (this is $os/$arch)."; exit 1
fi

# --- find a CUDA 13 runtime (for the preflight message + the launcher default) --------
find_cuda_lib() {
  local d
  for d in "${STEPPE_CUDA_LIB:-}" /usr/local/cuda/lib64 /usr/local/cuda-13*/lib64; do
    [ -n "$d" ] && [ -e "$d/libcudart.so.13" ] && { printf '%s\n' "$d"; return 0; }
  done
  if command -v ldconfig >/dev/null 2>&1; then
    d="$(ldconfig -p 2>/dev/null | awk '/libcudart\.so\.13/ {print $NF; exit}')"
    [ -n "$d" ] && { dirname "$d"; return 0; }
  fi
  return 1
}

if CUDA_LIB="$(find_cuda_lib)"; then
  say "✓ CUDA 13 runtime: $CUDA_LIB/libcudart.so.13"
else
  err "CUDA 13 runtime (libcudart.so.13) NOT found."
  err "  steppe is a CUDA-13 GPU product; a CUDA-12 box will not run the compute paths."
  err "  → install the CUDA 13 runtime, or set STEPPE_CUDA_LIB=/path/to/cuda/lib64 and re-run."
  err "  (downloading anyway — you can add CUDA 13 later; the GPU-free 'steppe-rds' f2"
  err "   converter in the Python wheel needs no CUDA.)"
fi

# --- GPU / driver (informational) -----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  [ -n "$gpu" ] && say "✓ GPU: $gpu"
else
  err "note: nvidia-smi not found — can't confirm a GPU/driver is present."
fi

# --- download the CLI -----------------------------------------------------------------
mkdir -p "$DIR"
url="https://github.com/$REPO/releases/download/$TAG/steppe"
say "↓ downloading steppe $TAG → $DIR/steppe.bin"
curl -fL --progress-bar -o "$DIR/steppe.bin" "$url"
chmod +x "$DIR/steppe.bin"

# --- write the CUDA-aware launcher ----------------------------------------------------
cat > "$DIR/steppe" <<'WRAP'
#!/bin/sh
# steppe launcher: put a CUDA 13 runtime on the loader path, or explain clearly why it can't.
self="$(CDPATH= cd "$(dirname "$0")" && pwd)"
lib=""
for d in "${STEPPE_CUDA_LIB:-}" /usr/local/cuda/lib64 /usr/local/cuda-13*/lib64; do
  [ -n "$d" ] && [ -e "$d/libcudart.so.13" ] && { lib="$d"; break; }
done
if [ -z "$lib" ] && command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -q 'libcudart\.so\.13' && lib="__inpath__"
fi
if [ -z "$lib" ]; then
  echo "steppe: CUDA 13 runtime not found (libcudart.so.13); steppe needs CUDA 13.x." >&2
  echo "  • CUDA 13 elsewhere?  set STEPPE_CUDA_LIB=/path/to/cuda/lib64" >&2
  echo "  • on CUDA 12?         steppe compute won't run here (the GPU-free 'steppe-rds'" >&2
  echo "                        f2 converter in the Python wheel still works)." >&2
  exit 127
fi
[ "$lib" != "__inpath__" ] && export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
exec "$self/steppe.bin" "$@"
WRAP
chmod +x "$DIR/steppe"

# --- stage the tiny quickstart example (a real 10-pop AADR f2 cache) -------------------
EXAMPLE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/steppe/example"
if [ "${STEPPE_NO_EXAMPLE:-}" != "1" ] \
   && curl -fL -s -o "/tmp/steppe_example.$$.tgz" \
        "https://github.com/$REPO/releases/download/$TAG/example_f2.tar.gz" 2>/dev/null; then
  rm -rf "$EXAMPLE_DIR" && mkdir -p "$EXAMPLE_DIR"
  if tar xzf "/tmp/steppe_example.$$.tgz" -C "$EXAMPLE_DIR" 2>/dev/null; then
    say "✓ example f2 cache → $EXAMPLE_DIR"
  else
    EXAMPLE_DIR=""
  fi
  rm -f "/tmp/steppe_example.$$.tgz"
else
  EXAMPLE_DIR=""   # asset absent / no network — skip the quickstart hint quietly
fi

# --- verify + next steps --------------------------------------------------------------
say ""
if "$DIR/steppe" --version >/dev/null 2>&1; then
  say "✓ installed: $("$DIR/steppe" --version) → $DIR/steppe"
else
  say "• installed to $DIR/steppe (couldn't run --version yet — see the CUDA note above)"
fi
# put $DIR on PATH persistently (opt out with STEPPE_NO_MODIFY_PATH=1)
case ":$PATH:" in
  *":$DIR:"*) ;;  # already on PATH — nothing to do
  *)
    added=""
    if [ "${STEPPE_NO_MODIFY_PATH:-}" != "1" ]; then
      case "${SHELL:-}" in
        */zsh)  rc="$HOME/.zshrc" ;;
        */bash) rc="$HOME/.bashrc" ;;
        *)      rc="$HOME/.profile" ;;
      esac
      if [ ! -f "$rc" ] || ! grep -qF "added by steppe install.sh" "$rc" 2>/dev/null; then
        printf '\n%s\n' "export PATH=\"$DIR:\$PATH\"  # added by steppe install.sh" >> "$rc" \
          && added="$rc"
      fi
    fi
    if [ -n "$added" ]; then
      say "• added $DIR to PATH in $added — open a new shell or: source $added"
    else
      say "• add $DIR to your PATH:  export PATH=\"$DIR:\$PATH\""
    fi
    ;;
esac
say ""
if [ -n "$EXAMPLE_DIR" ] && [ -f "$EXAMPLE_DIR/f2.bin" ]; then
  say "try it now — a real qpAdm fit on the bundled example (no data to download):"
  say "  steppe qpadm --f2-dir \"$EXAMPLE_DIR\" \\"
  say "    --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N \\"
  say "    --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian"
  say ""
fi
say "next:  steppe --help   ·   a full worked example (install → f2 → qpAdm):"
say "       # https://github.com/$REPO/blob/main/docs/examples"
