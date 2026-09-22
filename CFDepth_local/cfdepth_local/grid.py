import math
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from .contracts import GridSpec

R = 6378137.
DIRS = ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))

def grid_from_dataset(ds, row_offset=0, col_offset=0):
    return GridSpec(str(ds.crs), tuple(ds.transform)[:6], ds.height, ds.width, row_offset, col_offset)

def validate_alignment(mask, dem, tolerance=1e-5):
    if mask.count != 1 or dem.count != 1: raise ValueError('Single-band input required')
    if mask.shape != dem.shape or mask.crs != dem.crs: raise ValueError('Shape/CRS mismatch')
    if mask.crs != CRS.from_epsg(4326): raise ValueError('EPSG:4326 required')
    for d in (mask,dem):
        t=d.transform
        if not all(math.isfinite(v) for v in t) or t.b != 0 or t.d != 0 or t.a<=0 or t.e>=0:
            raise ValueError('North-up unrotated grid required')
        if d.bounds.bottom <= -90 or d.bounds.top >= 90: raise ValueError('Unsupported polar grid')
    inv=~mask.transform
    error=0.
    for col,row in ((0,0),(mask.width,0),(0,mask.height),(mask.width,mask.height)):
        x,y=inv @ (dem.transform @ (col,row))
        error=max(error,abs(x-col),abs(y-row))
    if error>tolerance: raise ValueError(f'Grid discrepancy {error} pixels exceeds {tolerance}')
    return grid_from_dataset(mask),error

def row_distances(grid, rows, dc, dr):
    a,_,_,_,e,f=grid.transform
    phi=f+e*(np.asarray(rows)+grid.row_offset+.5+dr*.5)
    dy=R*abs(e)*math.pi/180
    dx=R*abs(a)*math.pi/180*np.cos(np.deg2rad(phi))
    return np.hypot(dx*dc,dy*dr)

def row_weights(grid, lambda_c=1.):
    h=R*abs(grid.transform[4])*math.pi/180
    return np.column_stack([lambda_c*h*h/row_distances(grid,np.arange(grid.height),dc,dr)**2 for dc,dr in DIRS])
