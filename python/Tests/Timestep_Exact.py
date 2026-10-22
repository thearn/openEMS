"""
 Exact timestep (method 4) and diverged-run detection.

 A small PEC cavity with anisotropic, graded cells and a dielectric block. Method 4 computes the
 leapfrog limit of the discrete operator by Lanczos and uses 0.98 of it; the run must stay
 bounded. With TimeStepFactor 1.07 (1.049 x the exact limit) the run must diverge, and openEMS
 must raise instead of treating the non-finite energy as converged.
 With a GPU engine available, the device Lanczos (GPU_Backend::LanczosMaxEig) must find the same
 limit as the host one (OPENEMS_LANCZOS_CPU=1) within 0.5 %.

 (c) 2026 solvers-2026-09 campaign
"""
import os, re, sys, tempfile
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS


def build(factor=None, method=4):
    kw = dict(NrTS=6000, EndCriteria=1e-30)
    if factor is not None:
        kw['TimeStepFactor'] = factor
    FDTD = openEMS(**kw)
    FDTD.SetTimeStepMethod(method)
    FDTD.SetGaussExcite(5e9, 4e9)
    FDTD.SetBoundaryCond(['PEC'] * 6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(1e-3)
    mesh.AddLine('x', np.r_[np.linspace(-10, -2, 9), np.linspace(-1.8, 1.8, 19), np.linspace(2, 10, 9)])
    mesh.AddLine('y', np.r_[np.linspace(-10, -2, 9), np.linspace(-1.8, 1.8, 19), np.linspace(2, 10, 9)])
    mesh.AddLine('z', np.linspace(-6, 6, 25))
    CSX.AddMaterial('diel', epsilon=4.0).AddBox([-3, -3, -2], [3, 3, 2])
    exc = CSX.AddExcitation('exc', exc_type=0, exc_val=[0, 0, 1])
    exc.AddBox([0, 0, -0.5], [0, 0, 0.5])
    return FDTD


def run(FDTD, name):
    path = os.path.join(tempfile.gettempdir(), 'Timestep_Exact_' + name)
    FDTD.Run(path, cleanup=True, verbose=0)


run(build(method=4), 'stable')
print('method 4: bounded run completed')
try:
    run(build(factor=1.07, method=4), 'unstable')
except RuntimeError as e:
    assert 'diverged' in str(e), str(e)
    print('method 4 x 1.07: divergence detected')
else:
    raise AssertionError('FAIL: a timestep 4.9% above the exact limit did not diverge')


def exact_limit(engine, cpu):
    """the exact limit printed by the setup (fd 1 captured), or None"""
    os.environ['OPENEMS_LANCZOS_CPU'] = '1' if cpu else '0'
    path = os.path.join(tempfile.gettempdir(), 'Timestep_Exact_setup.txt')
    sys.stdout.flush()
    saved = os.dup(1)
    with open(path, 'w') as f:
        os.dup2(f.fileno(), 1)
        try:
            build(method=4).Run(os.path.join(tempfile.gettempdir(), 'Timestep_Exact_setup'), cleanup=True,
                                setup_only=True, verbose=1, engine=engine)
        except Exception as e:
            print('setup failed:', e)
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
    del os.environ['OPENEMS_LANCZOS_CPU']
    text = open(path).read()
    m = re.search(r'exact limit (\S+) s .*?(GPU Lanczos|Lanczos \d+ iterations)', text)
    return (float(m.group(1)), m.group(2).startswith('GPU')) if m else (None, False)


host, _ = exact_limit('gpu', cpu=True)
device, on_device = exact_limit('gpu', cpu=False)
if not on_device:
    print('no GPU Lanczos (no device backend): device check skipped')
else:
    assert host and abs(device/host-1) < 5e-3, f'FAIL: device limit {device} vs host {host}'
    print(f'GPU Lanczos limit {device:.6g} s, host {host:.6g} s ({100*(device/host-1):+.3f} %)')
print('PASS')
