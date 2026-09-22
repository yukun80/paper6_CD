"""Offline CFDepth, frozen v3.2.1 numerical rules."""
import os
import sys
from pathlib import Path

__version__ = "1.0.0"
BASELINE = "a88f1dd7897fc479e85ee2f6191506017a920457"
# Conda's native libraries need these when invoked without shell activation.
for key, folder in (("GDAL_DATA", "gdal"), ("PROJ_DATA", "proj")):
    location = Path(sys.prefix) / "Library" / "share" / folder
    if location.is_dir():
        os.environ.setdefault(key, str(location))
