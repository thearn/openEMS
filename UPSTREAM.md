# Upstream state and PR strategy for `cuda-perf`

Working notes, not for commit. Gathered 2026-09-23; every SHA and date below was
read live from the GitHub API or from `upstream/GPU_beta`, fetched as:

```bash
git fetch --no-tags git@github.com:SeanMollet/openEMS.git \
  GPU_beta:refs/remotes/upstream/GPU_beta
```

## Repository topology

- ours: `thearn/openEMS`, remote `origin`, working branch `cuda-perf`
- upstream: `SeanMollet/openEMS` (the fork parent of ours)
- above that: `thliebig/openEMS`, the original project

`cuda-perf` is based on `ad133bf`, the tip of `gpu_beta_cuda`.

`GPU_beta` is not an ancestor of ours and ours is not an ancestor of it. They
are two copies of the same series that part company at master:

```
merge-base cuda-perf upstream/GPU_beta = a8efa7d  (master, 2026-09-19)
rev-list --left-right --count             119 ours  |  121 theirs
git cherry upstream/GPU_beta cuda-perf    58 patch-equivalent  |  60 not
```

58 of our 119 commits already exist upstream as patch-equivalents with
different SHAs: `310ec51 openEMS --dry-run` is their `eb99ffe`, our
`47e9172 cuda: no Boost headers in the CUDA files` is their `bcb715e hip: no
Boost headers in the CUDA files`. Sean rebuilt the branch through the HIP
conversion instead of merging it. The 60 without an equivalent are almost all
the CUDA line, which upstream carries as `.hip` rewrites: `git cherry` compares
patch text, and a `CUDA_GridDim` to `HIP_GridDim` rename makes them look
unrelated.

## Branch status

| Branch | Upstream tip | Date | Relation to ours |
|---|---|---|---|
| `GPU_beta` | `a1e7885` | 2026-09-22 | **no new commits**; identical to `origin/GPU_beta` |
| `gpu_beta_cuda` | `ad133bf` | 2026-09-21 | identical to ours; our base |
| `master` | `65f8771` | 2026-09-22 | 7 commits ahead of `origin/master` |
| `metal-pr` | `98d02d1` | 2026-09-23 | not in our fork |
| `hooks-pr` | `61cbd7c` | 2026-09-19 | not in our fork |

**`gpu_beta_cuda` is the pre-HIP twin of `GPU_beta`, not abandoned work.**
`git merge-base --is-ancestor ad133bf upstream/GPU_beta` returns false, so
nothing here fast-forwards, but Sean still forward-ports across the two:
`b8da80c` on `GPU_beta` carries "(cherry picked from commit ad133bf)", which is
exactly our base commit. Active development is on `GPU_beta`; the CUDA line
receives cherry-picks rather than new work.

## The CUDA to HIP migration

`GPU_beta` has `FDTD/hip/*.hip` and no `FDTD/cuda/` at all. CUDA was not
dropped; HIP is a source-portability layer that compiles to CUDA on NVIDIA.
From `7cffa46` (2026-09-20), "FDTD: the GPU backend on HIP, for AMD as well as
NVIDIA":

> HIP compiles the same sources for both vendors: hipcc uses the ROCm compiler
> on AMD and hipcc on NVIDIA, so the backend no longer needs CUDA in its
> sources and openEMS runs on an AMD GPU.
>
> The translation itself is what hipify-perl does, the runtime calls and the
> headers; the kernels are unchanged, and there were no warp intrinsics or
> NVIDIA libraries to replace.

Measured at the migration: RTX 3090 free space 9371 to 9439 MCells/s, the same
within noise; `python/Tests/GPU_Engine.py` 18/18 with deviation 0.0 on both an
RTX 3090 and an MI300X. On NVIDIA, HIP is headers over the CUDA runtime, so
nothing is lost there.

The costs that followed, all in later commits:

- **wavefront width** (`656cacf`): CDNA wavefronts are 64, not 32. The fused
  tile now takes its shape as template arguments and the backend launches the
  instantiation matching the device `warpSize`. MI300X free space 22984 to
  24473 MCells/s (PML_8) and 30567 to 39483 (PEC).
- **determinism flags**: `--fmad=false` and `--ftz=true` reach hipcc, whereas
  the clang spellings would only reach the host compiler under it.
- **AMD failure mode** (`5d0c4c1`): a launch for an uncompiled architecture
  aborts the process instead of returning an error, so the backend now probes a
  kernel's attributes before committing, and falls back to the CPU reference.

## Where our six commits stand against `GPU_beta`

| Commit | Subject | Portability |
|---|---|---|
| `6d0750c` | `CalcUpdateCoefficients` hook, setup phase times | `operator.cpp`/`.h` **identical** to our base upstream; applies clean |
| `fc06954` | parallel x-range coefficient conversion | `operator_gpu.cpp` drift is 1 line (`AvailableCPUs` to `AvailableThreads`), outside our hunks; applies clean |
| `ee8e047` | fused-kernel coefficient-mode specialization | file gone; `gpu_backend_hip.hip` differs ~210 lines from our base even after renaming HIP to CUDA |
| `19ea80d` | merged/paired MUR post+apply | file gone; `hip_ext_mur_abc.hip` is a near-pure rename, 29 differing lines |
| `3b13658` | component-masked current fixup | same file as `ee8e047` |
| `46b7e6c` | operator identity guard | depends on our unmerged `5711922` (`RestartFDTD`); upstream `openems.cpp` also drifted 83+/16- |

Porting is therefore not moving onto a branch that discarded our work; it is
carrying three patches across a rename boundary Sean has already crossed once
with `hipify-perl`, and re-crossed by hand for the wavefront templating.

The `ee8e047` port is the hard one and is a **design question, not a rename**:
upstream already templates `update_fused` on tile shape for the wavefront, and
our change templates the same kernel on coefficient mode and region presence.
Combining both axes is four coefficient variants times two tile widths, eight
instantiations of the hottest kernel, with compile-time and instruction-cache
cost. Worth agreeing with Sean before writing it.

## Policy: DCO sign-off is now required

`65f8771` "doc: add AGENTS.md and state the DCO sign-off in the AI policy" adds
a Sign-off section to `AI_POLICY.md`:

> Every commit must carry a `Signed-off-by:` line certifying the Developer
> Certificate of Origin; `git commit -s` adds it. It is always the **last**
> line, with the AI tag directly above it. [...] Do not add `Co-Authored-By:`,
> session identifiers or links inserted by AI tooling, or other automatically
> generated trailers.

Our six commits carry the `Generated-by:` tags but **no `Signed-off-by:`**.
They need `git rebase --signoff` before any PR. Note also that the policy shows
one AI tag, singular; ours carry two (Claude Code and Codex), which may want a
word with Sean. `AGENTS.md` is new in that commit and has not been read yet.

## Test coverage

We added no tests. Sean's existing ones:

- `python/Tests/GPU_Engine.py` — the real acceptance test. Runs each case under
  `engine='basic'`, `'gpu-reference'` and `'gpu'` and compares every probe file
  and HDF5 dump: bit-identical for the CPU reference backend, `<1e-4` of peak
  for the device. Covers excitation, UPML, Mur, Lorentz, lumped RLC, conducting
  sheet, TF/SF, local absorber, steady-state, and cylindrical meshes with one
  and two multi-grid levels. Also asserts the requested backend was created
  rather than silently falling back.
- `.github/packaging/check_gpu_engine.py` — checks the shipped binary really
  contains the CUDA/Metal backend, works on GPU-less runners.

Both run from `.github/workflows/packages.yml`, **not** `ci.yml`. `GPU_Engine.py`
has no `test_` prefix, so the `unittest discover -p "test_*.py"` step in ci.yml
never collects it; the packaging jobs invoke it by path.

Coverage of what we landed: the default paths of all five performance commits
are exercised implicitly by `GPU_Engine.py` (wrong results would break the
comparisons). Not covered: the `OPENEMS_*=0` fallback paths that each commit
adds, and the operator identity guard / `RunReuse`, which nothing outside the
`.pyx` references.

Nothing in `cuda-perf` has been compiled from a clean checkout in this
environment. The campaign validated the final tree on a Tesla T4 with em-bench
fixtures (S3, A10, TFP-1), not with `GPU_Engine.py`, and the intermediate
commits were never built individually.

## Known defects  (both fixed 2026-09-23)

`openems.cpp` used `std::ostringstream` with no `#include <sstream>`, compiling
only through a transitive include. Fixed in `f05c4e1`.

`19ea80d`'s merged Mur post+apply deviates `3.1e-04` of peak from the basic
engine on `GPU_Engine.py`'s `mur` case. Defaulted off in `fa3e8ad`, replaced by
separate paired passes in `3abba41`, written up as `issues.md` item 8.

## Policy: work stays on our fork  (decided 2026-09-23)

**Do not open pull requests against `SeanMollet/openEMS` or
`thliebig/openEMS.`**  Asked for directly.  The PR strategy this file used to
carry is superseded; what stays useful is the topology above -- which branches
exist upstream, what the HIP migration did, and which of our commits would port
cleanly if that ever changes.

Push branches on `origin` freely; that is our own fork.

### Current state

- `cuda-perf` at `5554f01` (pushed): the six refinements, our `<sstream>` fix,
  the merged-Mur default flipped off, the ordering fix that replaced it, and
  the defect written up as `issues.md` item 8.
- `coeff-setup-pr` at `64598e7` (pushed, **no PR opened**): the two
  backend-agnostic commits -- the `CalcUpdateCoefficients` hook and the
  parallel coefficient conversion -- cherry-picked onto `upstream/GPU_beta`
  with DCO sign-off, and reworded from `CUDA:` to `FDTD:` since they sit in
  `Operator_GPU` rather than in a backend.  Both applied clean, exactly as the
  table above predicted.  The four files they touch compile against
  `GPU_beta`; the rest of that branch does not build here, wanting a newer
  CSXCAD than this environment has (`CSPropProbeBox::GetOverSampling`).
  Kept as a ready artifact, not as an intention.

### What was learned about the kernel commits

`19ea80d`'s merged Mur post+apply is **not answer-preserving** and has been
replaced on `cuda-perf` by separate paired passes (`3abba41`).  Anyone
revisiting the HIP port question should know the merge is unsound rather than
merely awkward to port: `Engine_GPU::VoltageHalfStep` orders all boundary
reads before all boundary writes across two loops, and non-opposite Mur faces
share edge cells.  See `issues.md` item 8.

The correct fix costs the speed.  Measured on three articles, the paired
version is 0.999x / 1.006x / 1.003x against the pre-`19ea80d` build -- the
1.056x and 1.095x came from the merge itself, i.e. from dropping a global
read of the boundary buffers, not from the launch-count reduction that pairing
recovers.

## Campaign documents

The third-campaign docs (13 `.md` files, `ledger.tsv`, `run_sweep.py`) were
committed as `85a56b7`, then dropped and force-pushed away to keep them out of
upstream. They survive in `../openems-mine/docs/autoresearch/cuda-third-20260923/`
and in this repo's reflog as `85a56b7` until garbage collection.
