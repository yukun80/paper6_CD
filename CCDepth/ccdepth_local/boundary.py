"""Boundary sampling: fixed first-pass peers; no cloud dependencies."""
import numpy as np
from .contracts import BoundaryData

def first_pass(z,labels,dry,boundary,slope,slope_valid,pair_radius,sigma_floor,mad_scale,min_samples,slope_scale,dispersion):
    h,w=z.shape
    out=np.zeros((h,w,8),np.float64)
    for r in range(h):
        for c in range(w):
            if not boundary[r,c]:continue
            cid=labels[r,c];wet=[];ds=[]
            for rr in range(max(0,r-pair_radius),min(h,r+pair_radius+1)):
                for cc in range(max(0,c-pair_radius),min(w,c+pair_radius+1)):
                    if labels[rr,cc]==cid:wet.append(z[rr,cc])
                    elif dry[rr,cc]:
                        touches=False
                        for ar in range(max(0,rr-1),min(h,rr+2)):
                            for ac in range(max(0,cc-1),min(w,cc+2)):
                                if labels[ar,ac]==cid:touches=True
                        if touches:ds.append(z[rr,cc])
            nw=len(wet);nd=len(ds);out[r,c,4]=nw;out[r,c,5]=nd
            reason=0
            if nw==0:reason|=1
            if nd==0:reason|=2
            if not slope_valid[r,c]:reason|=4
            if nw and nd:
                wa=np.array(wet);da=np.array(ds)
                wm=np.median(wa);dm=np.median(da)
                ws=max(sigma_floor,mad_scale*np.median(np.abs(wa-wm)))
                sigmad=max(sigma_floor,mad_scale*np.median(np.abs(da-dm)))
                lo=wm-ws;hi=dm+sigmad;out[r,c,0]=lo;out[r,c,1]=hi;out[r,c,2]=(lo+hi)*.5
                out[r,c,7]=ws+sigmad
                if lo>hi:reason|=8
                if reason==0:
                    variance=ws*ws+sigmad*sigmad
                    order=max(wm-dm,0.)
                    weight=min(1.,min(nw,nd)/min_samples)
                    weight*=1./(1.+(slope[r,c]/slope_scale)**2)*(1./(1.+variance/(dispersion*dispersion))*(1./(1.+order*order/variance)))
                    out[r,c,3]=weight
            out[r,c,6]=reason
    return out

def peer_pass(first,labels,boundary,peer_radius,high_weight,sigma_floor,mad_scale):
    h,w=labels.shape;beta=first[:,:,3].copy()
    for r in range(h):
        for c in range(w):
            if not boundary[r,c] or beta[r,c]==0:continue
            peers=[];cid=labels[r,c]
            for rr in range(max(0,r-peer_radius),min(h,r+peer_radius+1)):
                for cc in range(max(0,c-peer_radius),min(w,c+peer_radius+1)):
                    if (rr!=r or cc!=c) and labels[rr,cc]==cid and first[rr,cc,3]>=high_weight:
                        peers.append(first[rr,cc,2])
            if len(peers)>=3:
                a=np.array(peers);median=np.median(a);mad=np.median(np.abs(a-median))
                sigmap=max(sigma_floor,mad_scale*mad)
                excess=max(0.,abs(first[r,c,2]-median)-3.*sigmap)
                beta[r,c]*=1./(1.+(excess/first[r,c,7])**2)
    return beta

def boundary_arrays(inputs,domain,p,backend=None):
    from .terrain import compute_slope
    slope,sv=compute_slope(inputs.dem,inputs.dem_valid,inputs.grid)
    f1,f2=(first_pass,peer_pass) if backend is None else backend
    first=f1(inputs.dem,domain.component_id,domain.dry,domain.boundary,slope,sv,p.pairRadius,p.sigmaFloor,p.madScale,p.minSamplesPerSide,p.slopeScaleDeg,p.dispersionScaleMeters)
    beta=f2(first,domain.component_id,domain.boundary,p.peerRadius,p.highWeight,p.sigmaFloor,p.madScale)
    return first,beta

def component_boundary(first,beta,topology,dem,hard,p):
    rr,cc=topology.rows,topology.cols;b=beta[rr,cc];a=first[rr,cc]
    high=b>=p.highWeight
    eligible=bool(high.sum()>=p.minAnchors and max(np.ptp(rr[high]),np.ptp(cc[high]))>=p.minSpanPixels) if high.any() else False
    initial=np.zeros(len(rr),np.float64)
    if eligible:
        s0=float(np.sum(b*a[:,2])/np.sum(b));initial[:]=s0
        initial[hard]=np.maximum(s0,dem[hard]+p.minDepth)
    return BoundaryData(a[:,0].copy(),a[:,1].copy(),a[:,2].copy(),a[:,3].copy(),b.copy(),a[:,4].astype(np.int16),a[:,5].astype(np.int16),a[:,6].astype(np.uint8),eligible,initial)
