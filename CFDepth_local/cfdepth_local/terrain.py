import numpy as np
from .grid import row_distances

def compute_slope(dem,valid,grid):
    slope=np.zeros(dem.shape,np.float64);sv=np.zeros(dem.shape,bool)
    sv[1:-1,1:-1]=valid[1:-1,1:-1]&valid[:-2,1:-1]&valid[2:,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]
    dx=row_distances(grid,np.arange(grid.height),1,0)[:,None]
    dy=row_distances(grid,np.arange(grid.height),0,1)[:,None]
    gx=(dem[1:-1,2:]-dem[1:-1,:-2])/(2*dx[1:-1])
    gy=(dem[2:,1:-1]-dem[:-2,1:-1])/(2*dy[1:-1])
    slope[1:-1,1:-1]=np.degrees(np.arctan(np.hypot(gx,gy)))
    slope[~sv]=np.nan
    return slope,sv
