#!/usr/bin/env bash
# Build this checkout into build/install and point a Python environment's
# openEMS package at it (editable, so python/openEMS is imported from here).
#
#   scripts/build-local.sh [PYTHON]
#
# PYTHON        interpreter whose environment gets the bindings
#               (default: $HOME/.venvs/antenna-foundry/bin/python)
# CONDA_PREFIX_OPENEMS  conda environment providing the compilers, CUDA,
#               CSXCAD, VTK, HDF5 and Boost
#               (default: $HOME/miniforge-speedup/envs/openems-gpu)
# CUDA_ARCHS    default "75;80;86;89" (T4, A100, RTX 30, RTX 40)
#
# Configure runs every time so the embedded `git describe` version matches
# HEAD; antenna-foundry refuses a build whose version differs from the repo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-$HOME/.venvs/antenna-foundry/bin/python}"
E="${CONDA_PREFIX_OPENEMS:-$HOME/miniforge-speedup/envs/openems-gpu}"
ARCHS="${CUDA_ARCHS:-75;80;86;89}"
PREFIX="$REPO/build/install"
export PATH="$E/bin:$PATH"

cmake -S "$REPO" -B "$REPO/build" -G Ninja -Wno-dev \
  -DCMAKE_MAKE_PROGRAM="$E/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  -DCMAKE_C_COMPILER="$E/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$E/bin/x86_64-conda-linux-gnu-g++" \
  -DCMAKE_CUDA_COMPILER="$E/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$E/bin/x86_64-conda-linux-gnu-g++" \
  -DENABLE_CUDA=ON -DENABLE_FLUSH_TO_ZERO=ON -DENABLE_RPATH=ON \
  -DCMAKE_PREFIX_PATH="$E" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_RPATH="$PREFIX/lib;$E/lib" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$E/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,$E/lib" >/dev/null
ninja -C "$REPO/build" install >/dev/null

# The conda interpreter's LDSHARED puts $E/lib first in the RPATH, which would
# load the conda libopenEMS instead of this build; CPATH keeps the conda
# openEMS headers behind build/install/include.
cd "$REPO/python"
OPENEMS_INSTALL_PATH="$PREFIX" OPENEMS_NOSCM=1 \
CC="$E/bin/x86_64-conda-linux-gnu-gcc" CXX="$E/bin/x86_64-conda-linux-gnu-g++" \
LDSHARED="$E/bin/x86_64-conda-linux-gnu-gcc -shared" \
LDCXXSHARED="$E/bin/x86_64-conda-linux-gnu-g++ -shared" \
CPATH="$E/include" \
LDFLAGS="-Wl,-rpath,$PREFIX/lib -Wl,-rpath,$E/lib -L$PREFIX/lib -L$E/lib -Wl,-rpath-link,$E/lib" \
  "$PYTHON" -m pip install -q --no-build-isolation --no-deps \
  -e . --config-settings editable_mode=compat

cd /
"$PYTHON" - "$REPO" <<'EOF'
import sys
from pathlib import Path
import openEMS
from openEMS import openEMS as _solver  # noqa: F401  loads libopenEMS
repo = Path(sys.argv[1]).resolve()
libs = {Path(l.split()[-1]) for l in open('/proc/self/maps') if 'libopenEMS.so' in l}
assert Path(openEMS.__file__).resolve().is_relative_to(repo), openEMS.__file__
assert len(libs) == 1 and libs.pop().resolve().is_relative_to(repo / 'build/install/lib'), libs
print(f'openEMS from {repo}')
EOF
strings "$PREFIX/lib/libopenEMS.so.0" | grep -m1 -E '^v[0-9]+\.[0-9]+\.[0-9]+'
