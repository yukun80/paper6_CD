from pathlib import Path
import os
import numpy as np
import rasterio
from affine import Affine
from .contracts import RasterInputs, GridSpec
from .grid import validate_alignment
from .provenance import file_hash

def valid_dem(values, mask):
    return (mask>0)&np.isfinite(values)&(np.abs(values)<1e11)

def read_inputs(mask_path,dem_path,config=None,window=None,hash_inputs=True):
    with rasterio.open(mask_path) as m, rasterio.open(dem_path) as d:
        grid,error=validate_alignment(m,d)
        a=m.read(1,window=window); observed=m.read_masks(1,window=window)>0
        if np.any(observed & (a!=0)&(a!=1)): raise ValueError('Valid mask values must be 0/1')
        z=d.read(1,window=window).astype(np.float64); v=valid_dem(z,d.read_masks(1,window=window))
        if window is not None:
            # Original affine plus offsets: parity and latitude remain global.
            grid=GridSpec(grid.crs,grid.transform,*z.shape,int(window.row_off),int(window.col_off))
    return RasterInputs(z,a==1,observed,v,grid,
                        {'mask':file_hash(mask_path),'dem':file_hash(dem_path)} if hash_inputs else {})

def write_raster(path,values,valid_mask,grid,dtype='float32',nodata=-9999):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if values.shape!=grid.shape or valid_mask.shape!=grid.shape: raise ValueError('Output shape mismatch')
    temp=path.with_name(path.name+'.tmp.tif')
    transform=Affine(*grid.transform) @ Affine.translation(grid.col_offset,grid.row_offset)
    profile=dict(driver='GTiff',height=grid.height,width=grid.width,count=1,dtype=dtype,crs=grid.crs,
                 transform=transform,nodata=nodata,compress='deflate',tiled=True,BIGTIFF='IF_SAFER')
    with rasterio.open(temp,'w',**profile) as out:
        out.write(np.where(valid_mask,values,nodata).astype(dtype),1)
    with rasterio.open(temp) as check:
        if check.shape!=grid.shape or check.transform!=transform: raise RuntimeError('Write validation failed')
    os.replace(temp,path)
