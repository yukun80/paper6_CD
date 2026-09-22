"""Conda native DLL lookup for direct python.exe launches on Windows."""
import os
import sys
from pathlib import Path

prefix=Path(sys.prefix)
native=prefix/'Library'/'bin'
if os.name=='nt' and native.is_dir():
    # Keep native failures in exit codes/logs instead of blocking unattended runs
    # behind a Windows error dialog; this does not turn failures into success.
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    parts=os.environ.get('PATH','').split(os.pathsep)
    if str(native).casefold() not in {p.casefold() for p in parts}:
        os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
for name,folder in [('GDAL_DATA','gdal'),('PROJ_DATA','proj')]:
    path=prefix/'Library'/'share'/folder
    if path.is_dir():os.environ.setdefault(name,str(path))
