import numpy as np
from .coordinate import coordinate_minimum

def diagnostic_loop(S,neighbors,weights,rows,degree,lower,upper,mid,beta,terrain,hard,lambda_b,lambda_t,mu,minimum):
    primary=0.;midterm=0.;norm=0.;residual=0.;violation=0.;bad=0
    for i in range(len(S)):
        ns=0.;continuity=0.
        for k in range(8):
            j=neighbors[i,k]
            if j>=0:
                w=weights[rows[i],k];ns+=w*S[j];continuity+=.5*w*(S[i]-S[j])**2
        b=beta[i];soft=0. if hard[i] else lambda_t
        lo=lower[i] if b>0 else 0.;hi=upper[i] if b>0 else 0.;md=mid[i] if b>0 else 0.
        distance=max(0.,max(lo-S[i],S[i]-hi))
        e=continuity+lambda_b*b*distance**2+soft*max(terrain[i]-S[i],0.)**2
        em=b*(S[i]-md)**2
        opt=minimum(degree[i],ns,lambda_b*b,lo,hi,soft,terrain[i],hard[i],mu*b,md)
        if not(np.isfinite(S[i]) and abs(S[i])<1e11 and np.isfinite(opt) and abs(opt)<1e11 and np.isfinite(e) and abs(e)<1e11 and np.isfinite(e+mu*em) and abs(e+mu*em)<1e11):bad+=1
        primary+=e;midterm+=em;norm+=.5*degree[i]+lambda_b*b+soft
        residual=max(residual,abs(opt-S[i]))
        if hard[i]:violation=max(violation,max(terrain[i]-S[i],0.))
    q=max(norm,1e-30)
    return primary,midterm,norm,primary/q,midterm/q,(primary+mu*midterm)/q,residual,violation,bad

def model_args(model,p):
    t=model.topology;b=model.boundary
    return (t.neighbors,t.weights,t.rows,t.degree,b.lower,b.upper,b.mid,b.beta,model.dem+p.minDepth,model.hard,p.lambdaB,p.lambdaT)

def diagnostics(S,model,p,mu=0.,backend=None):
    loop,minimum=(diagnostic_loop,coordinate_minimum) if backend is None else backend
    vals=loop(S,*model_args(model,p),mu,minimum)
    return dict(zip(('J0','Jm','Q','primary','mid','total','residual','hard','bad'),map(float,vals)))

def energy_terms(S,model,p,mu=0.):
    return diagnostics(S,model,p,mu)
