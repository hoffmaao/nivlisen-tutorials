#!/usr/bin/env bash
#
# Build the Fill-Spill-Merge meltwater-routing CLIs (fsm_wrapper, fsm_batch) for
# the Nivlisen tutorials, WITHOUT Docker.
#
# Only notebook 03 (03-melt-flux) needs these; it routes surface meltwater with
# Fill-Spill-Merge (Barnes et al., 2020). The Docker image compiles them for you
# (see the Dockerfile's Fill-Spill-Merge stage); this script does the same on
# your own machine so the notebook's route_fsm() / FSMRouter find them on PATH.
# Notebooks 00, 01, 02 and 04 run fine without it.
#
# It is self-contained. It clones the header-only upstream libraries at a pinned
# commit into a throwaway temp directory, compiles our two small CLIs against
# them with your default C++ compiler, installs the binaries somewhere on your
# PATH, then deletes the clone. Unlike the upstream repo's own build it needs no
# CMake, GDAL, OpenMP or Homebrew LLVM: our CLIs are header-only and do raw
# binary I/O, so a single `c++ -std=c++17` compile is all it takes. Stock Apple
# clang (macOS) or g++ (Linux) is enough.
#
# Usage:
#     fsm/build_fsm.sh [DEST_DIR]
#
#   DEST_DIR  where to install the two binaries. Default: the active Python
#             virtualenv's bin ($VIRTUAL_ENV/bin) if a venv is active, so the
#             binaries land on PATH automatically for the Jupyter kernel that
#             runs the notebooks; otherwise <repo>/fsm/bin, and the script prints
#             the one PATH line you need to add.
#
# Environment overrides:
#     CXX=...          C++ compiler to use (default: c++)
#     FSM_COMMIT=...   upstream commit to build against (default below)
#
set -euo pipefail

# Keep this in lockstep with the Dockerfile's FSM_COMMIT so the host build and
# the container build track the same upstream Fill-Spill-Merge revision.
FSM_COMMIT="${FSM_COMMIT:-1c499ea475c09b9f4c5da74ee5cc995de169db63}"
CXX="${CXX:-c++}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the fsm/ directory

# --- Where do the binaries go? -----------------------------------------------
if [[ $# -ge 1 ]]; then
    dest="$1"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    dest="$VIRTUAL_ENV/bin"
else
    dest="$here/bin"
fi
mkdir -p "$dest"

# --- Toolchain check ---------------------------------------------------------
if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: C++ compiler '$CXX' not found." >&2
    echo "       macOS:         xcode-select --install   (gives you clang)" >&2
    echo "       Debian/Ubuntu: sudo apt install g++" >&2
    echo "       or set CXX to your compiler, e.g. CXX=g++-13 fsm/build_fsm.sh" >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "error: git not found; it is needed to fetch the Fill-Spill-Merge sources." >&2
    exit 1
fi

# --- Clone the upstream headers at the pinned commit (temporary) -------------
src="$(mktemp -d)"
trap 'rm -rf "$src"' EXIT

echo "==> Fetching Fill-Spill-Merge headers @ ${FSM_COMMIT:0:12} (one-time, ~a minute) ..."
git clone --quiet https://github.com/r-barnes/Barnes2020-FillSpillMerge.git "$src"
git -C "$src" checkout --quiet "$FSM_COMMIT"
git -C "$src" submodule update --init --recursive --quiet

# --- Compile both CLIs -------------------------------------------------------
echo "==> Compiling fsm_wrapper and fsm_batch with '$CXX' ..."
for tool in fsm_wrapper fsm_batch; do
    "$CXX" -O2 -std=c++17 \
        -I"$src/include" \
        -I"$src/submodules/dephier/include" \
        -I"$src/submodules/dephier/submodules/richdem/include" \
        -o "$dest/$tool" "$here/$tool.cpp"
    echo "    built $dest/$tool"
done

# --- Smoke check: each CLI should at least execute ---------------------------
# Run with no arguments; each prints a usage message and exits 1. Any other exit
# status (e.g. 126 "cannot execute binary" from an architecture mismatch) means
# the build is broken, so surface it now rather than deep inside notebook 03.
for tool in fsm_wrapper fsm_batch; do
    rc=0; "$dest/$tool" >/dev/null 2>&1 || rc=$?
    if [[ "$rc" -ne 1 ]]; then
        echo "warning: $dest/$tool did not run as expected (exit $rc)." >&2
    fi
done

# --- Tell the student about PATH ---------------------------------------------
echo
echo "==> Installed fsm_wrapper and fsm_batch into: $dest"
case ":${PATH}:" in
    *":${dest}:"*)
        echo "    $dest is already on your PATH, so notebook 03 will find them."
        ;;
    *)
        echo "    NOTE: $dest is NOT on your PATH. Add it before launching Jupyter:"
        echo
        echo "        export PATH=\"$dest:\$PATH\""
        echo
        echo "    (put that line in your shell profile to make it permanent)."
        ;;
esac
