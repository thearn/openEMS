# CUDA GPU engine: optimization notes

Performance work on the CUDA backend of the GPU engine (`--engine=gpu`,
`FDTD/cuda/`): what was done, what it gained, and what is left. Many of the
changes are ports of the Metal work (see `Optimizations-Metal.md`).

AI disclosure: measured, implemented and written up with Claude Opus 5 (Claude Code).

## Benchmark

Release build, RTX 2080 Ti (616 GB/s), Xeon E5-2673 v4 host with 80 threads.
The mesh is free space, 200^3 = 8 million cells, with an 8-cell PML on all
sides, a soft dipole source, one field probe and 4000 timesteps. The graded
variant uses mesh spacings that vary with a period of 97 lines in all three
directions (1.47 million distinct coefficient sets). Profiles were taken with
`nvprof`.

## Results

| Step (commit) | Uniform mesh | PEC walls instead of PML | Graded mesh |
|---|---|---|---|
| Baseline | 1211 MCells/s | 1558 | - |
| End-criterion energy on the GPU | 1292 | - | - |
| Fields read from the device on demand | 2456 | 4544 | - |
| Compressed update coefficients | 2990 | 6766 | - |
| UPML fused with the main updates, compressed UPML coefficients | 4418 | - | 3480 |
| UPML regions along z in the main kernels | **5210** | - | **3988** |

For comparison, the multithreaded CPU engine runs the benchmark at 186 MCells/s
on the same machine (80 threads).

Every step is bit-exact: `python/Tests/GPU_Engine.py` shows the same
deviations before and after. The coefficient steps were also run with each
layout forced (full arrays, 16 bit and 32 bit indices). All physics tests in
`python/Tests/` pass on the CUDA engine.

### What was done

1. **Fields read from the device on demand** (`Engine_GPU`, shared with all
   backends without shared memory). This was the largest item.
   - Before, both fields were copied to the host after every batch: 47 % of
     the GPU time, since the benchmark's batches are only ~6 timesteps long.
   - Now the host copy is marked out of date at the end of a batch.
     `GetVolt`/`GetCurr` read the z-line of a value from the device
     (`GPU_Backend::DownloadRange()`), which is what probes need.
   - After 256 lines in a batch, the whole field is copied once, which is what
     dumps need. A batch that needed the whole field copies it at its first
     read in the next batch.
2. **End-criterion energy on the GPU.** This is the Metal kernel: sums in float
   along x per (y,z) line, and the host sums the lines in double. Without it,
   the energy check alone would force a full copy.
3. **Compressed update coefficients.** A 16 or 32 bit set index per node and the
   distinct sets. The set search is shared with Metal (`gpu_coeff_sets.cpp`).
4. **Fused and compressed UPML**, as on Metal. The fusion conditions
   (`GPU_UPMLFusionBox()`) are shared by both backends.
5. **UPML regions along z in the main kernels.**
   - Those regions are only 9 nodes deep along z, the contiguous direction.
     Their rows touch 36 bytes of the fields, which wastes most of every memory
     transaction: 124 us per launch against 49 us for the other regions.
   - They cover exactly the x/y range of the main updates, so the main kernels
     now run along full z lines and do the fused UPML update for those nodes.
   - This could be ported to Metal too, where the same layout effect should
     exist.

## Where the time goes now

- The main kernels take ~72 % of the GPU time and the x/y UPML regions ~24 %,
  both at ~450-510 GB/s. That is close to what the card delivers in practice.
- Gaps between the kernels and host work are negligible for this mesh size.
- Further gains need fewer bytes per timestep, e.g. temporal blocking
  (several timesteps per pass over a tile). That is a large change.

## Remaining ideas

- **Kernel launch overhead** (CUDA Graphs, fewer launches). A timestep is ~11
  launches; for 8 million cells that is ~2 % of the time. It matters for small
  meshes of ~100k cells, where the GPU work per timestep is only tens of
  microseconds.
- **Multi-grid levels on separate streams.** All grids of a cylindrical
  multi-grid share one stream. The sub-grids are small and do not fill the GPU.
- **FMA contraction** (`--fmad=false` keeps the results exact). This is probably
  irrelevant, since the kernels are memory bound.
- **Setup** (not a priority: real jobs are dominated by the computation).
  - At 200^3 the host setup takes ~17 s on this Xeon, against ~3.7 s on an M5
    Max: `Calc_EC`, `CalcTimestep`, the UPML build and the per-node
    coefficients.
  - `Calc_EC` scales poorly beyond ~10 threads on this machine: 1.7 s with 10
    threads, 4.9 s with 80.

## Antenna models (2026-09 speedups campaign)

A second round driven by two antenna models from antenna-foundry (TFP-1 and
Pagoda-3: 19 to 44 million cells, UPML, conducting sheets with a 4-order ADE,
coax ports), on an RTX 4060 (8 GB, sm_89) and a Colab A100 (40 GB, sm_80).
The campaign record (plan, per-experiment ledger, report) is in antenna-foundry
`docs/autoresearch/speedups-2026-09/`. Stepping time per case, fork before the
campaign (e337998) and after (this branch), same inputs:

| Case | RTX 4060 before | after | A100 before | after |
|---|---:|---:|---:|---:|
| Pagoda fast (19 M cells) | 157 s | 109 s | 36.2 s | 33.4 s |
| TFP-1 fast (44 M cells) | 683 s | 364 s | 150 s | 117 s |

(The "after" runs also use a wider excitation pulse chosen in antenna-foundry,
which cuts the TFP-1 step count by 5.8%.)

Setup and correctness:

- The conducting-sheet scan read sheet shapes that another thread was
  rewriting (`CSPrimBox::GetBoundBox`), so parallel setups occasionally built a
  different operator. Shapes are now captured before the threads start.
- `OPENEMS_OPERATOR_CHECKSUM=1` covers the excitation, Mur, TF/SF and lumped
  RLC extensions as well as the main operator.
- Material lookups: candidate primitive lists by const reference, one priority
  lookup per sample point for paired properties, and an exact polygon inside
  test from a y-indexed edge list (`OPENEMS_POLYGON_INDEX=0` disables it,
  `OPENEMS_POLYGON_INDEX_VERIFY=1` checks it against the old test).
- Excitation rows without candidates are skipped; coefficient sets are found
  in parallel.
- The PEC dump (`DumpPEC2File`) includes conducting-sheet edges (a `sheet` cell
  array), is collected per plane in parallel, written uncompressed, and appears
  atomically (written to a temporary name, then renamed).
- `ConductingSheetMaxFreq` sets the sheet model's fit band independently of the
  excitation.

Runs:

- The engine is released after `RunFDTD`; `OPENEMS_DEVICE_LOCK=PATH` serializes
  the device between processes (flock) while another process builds its
  operator.
- `RunReuse` accepts a changed soft excitation amplitude (another driven port):
  the amplitudes are not part of the operator identity and the excitation is
  rebuilt.

CUDA:

- `OPENEMS_CUDA_FUSION_REPORT=1` explains why UPML regions stay out of the
  fused kernel. The current fix-up pass keeps the regions in the kernel when its
  nodes lie in them (double-buffered current flux).
- Dispersive (ADE) orders at shared positions run in one launch; inactive
  components are skipped.
- The fused step falls back to separate E and H updates when it would not fit in
  device memory with a reserve (`OPENEMS_CUDA_MEMORY_RESERVE_MB`, default 512).
- The fused kernel with UPML regions is specialized on the coefficient mode.
- One conducting-sheet voltage ADE group can be corrected inside the fused
  kernel instead of a separate launch plus a current fix-up. Both give identical
  fields. It is a template parameter: an untaken runtime check alone made the
  A100 18% slower. In the kernel, TFP-1 fast steps 5% faster on the RTX 4060 and
  17% slower on the A100, so it is on by default only for compute capability 8.9
  (`OPENEMS_CUDA_FUSED_ADE=1/0` forces it).

Measured and not pursued: CUDA graphs and batched probe gathers (host gaps are
at most 3%), other fused tile widths (within 1%), a CUDA NF2FF (the separable
CPU far field takes 3 to 5 s), temporal blocking (the steps between half-steps
carry the excitation, sheet ADE and fix-ups).

## Known limits (not performance, but related)

- Float precision only.
- 32-bit indexing in the kernels: meshes up to ~1.4 billion cells.
- Device memory: ~24 bytes per node for the fields plus the compressed
  coefficients, or 72 bytes per node with the full arrays. The host keeps a
  pinned mirror of the fields (24 bytes per node).
- `CMAKE_CUDA_ARCHITECTURES` is set to `native`, so a build only runs on GPUs of
  the build machine's architecture. Packaged builds need an explicit list (for
  example `75;80;86;89;90`) plus PTX for newer GPUs.
- The host fallback, used only for an extension without a CUDA implementation,
  moves both fields across PCIe twice per half-step.
