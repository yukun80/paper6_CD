import numpy as np
from .contracts import TopologyData
from .grid import DIRS,row_weights

def build_topology(component,grid,lambda_c=1.,allocator=None):
    rows,cols=np.nonzero(component);rows=rows.astype(np.int32);cols=cols.astype(np.int32)
    n=len(rows);alloc=allocator or (lambda name,shape,dtype:np.empty(shape,dtype=dtype))
    index=alloc('index',component.shape,np.int32);index[:]=-1;index[rows,cols]=np.arange(n,dtype=np.int32)
    neighbors=alloc('neighbors',(n,8),np.int32);neighbors[:]=-1
    weights=row_weights(grid,lambda_c);degree=np.zeros(n,np.float64)
    for k,(dc,dr) in enumerate(DIRS):
        rr=rows+dr;cc=cols+dc;inside=(rr>=0)&(rr<grid.height)&(cc>=0)&(cc<grid.width)
        neighbors[inside,k]=index[rr[inside],cc[inside]]
        degree+=np.where(neighbors[:,k]>=0,weights[rows,k],0)
    color=((cols+grid.col_offset)%2+2*((rows+grid.row_offset)%2)).astype(np.uint8)
    return TopologyData(rows,cols,neighbors,weights,degree,color,component.shape)
