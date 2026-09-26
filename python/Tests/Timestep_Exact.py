"""
 Exact timestep (method 4) and diverged-run detection.

 A small PEC cavity with anisotropic, graded cells and a dielectric block. Method 4 computes the
 leapfrog limit of the discrete operator by Lanczos and uses 0.98 of it; the run must stay
 bounded. With TimeStepFactor 1.07 (1.049 x the exact limit) the run must diverge, and openEMS
 must raise instead of treating the non-finite energy as converged.

 (c) 2026 solvers-2026-09 campaign
"""
import os, tempfile
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
print('PASS')
