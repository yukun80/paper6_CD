import numpy as np
from .grid import row_distances

PRODUCTS={
 'CFDepth_WSE.tif':('float32',-9999),
 'CFDepth_depth_solved.tif':('float32',-9999),
 'CFDepth.tif':('float32',-9999),
 'CFDepth_depth_signed.tif':('float32',-9999),
 'CFDepth_WSE_gradient_solved.tif':('float32',-9999),
 'CFDepth_gradient_directions.tif':('uint8',255),
 'CFDepth_status.tif':('int8',-1),
 'CFDepth_QA.tif':('uint16',65535),
}

def gradient_loop(S,neighbors,dx,dy):
    n=len(S);g=np.zeros(n,np.float64);directions=np.zeros(n,np.uint8)
    for i in range(n):
        for axis in range(2):
            plus=neighbors[i,4 if axis==0 else 6];minus=neighbors[i,3 if axis==0 else 1]
            distance=dx[i] if axis==0 else dy[i]
            if plus>=0 or minus>=0:
                if plus>=0 and minus>=0:v=(S[plus]-S[minus])/(2*distance)
                elif plus>=0:v=(S[plus]-S[i])/distance
                else:v=(S[i]-S[minus])/distance
                g[i]+=v*v;directions[i]|=1<<axis
        g[i]=np.sqrt(g[i])
    return g,directions

def build_products(result,model,grid,p,gradient_backend=None):
    n=len(model.dem);success=result.state.status in (3,4) and result.audit['passed']
    out={name:np.full(n,nodata,dtype=dtype) for name,(dtype,nodata) in PRODUCTS.items()}
    out['CFDepth_status.tif'][:]=result.state.status
    qa=np.zeros(n,np.uint16)
    if not success:
        qa[:]=4 if result.state.status==0 else 8
    else:
        signed=result.S-model.dem;legacy=signed>p.minDepth
        out['CFDepth_WSE.tif'][:]=result.S
        out['CFDepth_depth_solved.tif'][:]=np.maximum(signed,0)
        out['CFDepth_depth_signed.tif'][:]=signed
        out['CFDepth.tif'][legacy]=signed[legacy]
        t=model.topology
        dx=row_distances(grid,t.rows,1,0);dy=row_distances(grid,t.rows,0,1)
        grad,directions=(gradient_backend or gradient_loop)(result.S,t.neighbors,dx,dy)
        out['CFDepth_WSE_gradient_solved.tif'][directions>0]=grad[directions>0]
        out['CFDepth_gradient_directions.tif'][:]=directions
        qa[:]=16
        if result.state.status==4:qa|=32
        qa[(signed>=0)&(signed<=p.minDepth)]|=64
        qa[signed<0]|=128
        qa[(directions==1)|(directions==2)]|=256
        qa[model.hard&(np.abs(result.S-(model.dem+p.minDepth))<=p.hardTolerance)]|=512
    out['CFDepth_QA.tif']=qa
    return out
