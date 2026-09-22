# Tested Windows runtime

The authorized environment is `figure11/cenv` (Python 3.11). `before.json`, `after.json` and `changes.json` record the actual package changes; NumPy/SciPy/Rasterio/Numba were reused. `pip check` passed after installation.

Reproduction records:

- `conda-explicit.txt`: exact installed conda package URLs.
- `conda_records.json`: complete installed conda metadata.
- `requirements.lock`: actual installed Python distribution versions, including pre-existing packages; not a claim that all are production dependencies.
- `cfdepth_runtime_bootstrap.py`: direct-interpreter native DLL and GDAL/PROJ data initialization.

After rebuilding a compatible `figure11/cenv`, install the local package without changing locked numeric dependencies and install its environment bootstrap:

```powershell
& ../cenv/python.exe -m pip install --no-deps --no-build-isolation -e .
& ../cenv/python.exe scripts/bootstrap_environment.py
& ../cenv/python.exe -m pytest tests/test_m00_environment.py -q
& ../cenv/python.exe -m pip check
```

Run commands above from CFDepth_local. No global Windows environment variables or unrelated conda environments are changed. The bootstrap suppresses modal native crash dialogs while retaining nonzero exit codes; tests and stage reports still fail on errors. Diagnosis and minimal reproduction are retained in native_runtime_diagnosis.md.
