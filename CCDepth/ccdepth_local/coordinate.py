"""Six regions in frozen side/below order, strict less-than tie selection."""
def coordinate_minimum(degree,neighbor_sum,boundary,lower,upper,soft,terrain,hard,mid_weight,mid):
    # Inactive constraints must not leak NaN through zero multiplication.
    if boundary==0 and mid_weight==0: lower=0.;upper=0.;mid=0.
    mean=neighbor_sum/max(degree,1e-30);best=1e30;answer=0.
    for side in (-1,0,1):
        for below in (0,1):
            lo=upper if side==1 else -1e12
            hi=lower if side==-1 else 1e12
            if side==0:lo=lower;hi=upper
            lo=max(lo,-1e12 if below else terrain)
            hi=min(hi,terrain if below else 1e12)
            lo=max(lo,terrain if hard else -1e12)
            bw=0. if side==0 else boundary
            target=lower if side==-1 else upper
            tw=soft if below else 0.
            denominator=max((degree+bw)+(tw+mid_weight),1e-30)
            num=(neighbor_sum+bw*target)+(tw*terrain+mid_weight*mid)
            s=max(lo,min(hi,num/denominator))
            distance=max(0.,max(lower-s,s-upper))
            energy=degree*(s-mean)**2+(boundary*distance**2+(soft*max(0.,terrain-s)**2+mid_weight*(s-mid)**2))
            if lo<=hi and energy<best:answer=s;best=energy
    return answer
