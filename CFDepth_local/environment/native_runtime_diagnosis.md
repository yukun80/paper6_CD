# Windows native runtime diagnosis

Symptom: scipy.optimize BFGS crashed with Windows exception 0xc06d007f at np.dot(A1,np.dot(Hk,A2)).
Minimal reproduction: python -c "import numpy as np;print(np.dot(np.eye(3),np.eye(3)))" exited 1; matrix-vector np.dot succeeded.

Controlled probes:
- Adding only os.add_dll_directory(cenv/Library/bin): still exit 1.
- Prepending only cenv/Library/bin to the child process PATH: exit 0, correct identity matrix.

Root cause: direct interpreter launch did not provide the conda native DLL search PATH needed by delay-loaded MKL components. No NumPy/SciPy versions changed. A cenv-only .pth bootstrap initializes the process PATH and GDAL/PROJ data locations. The system/user PATH is unchanged.

Regression: tests/test_m00_environment.py invokes a fresh direct interpreter and checks matrix multiplication plus an actual BFGS solve. The original independent graph test is retained.

The production process was stopped after prepared completion and while processing small components; its committed journal and checkpoint remain resumable. Algorithm and prepared hashes are unchanged by this environment-only correction.
