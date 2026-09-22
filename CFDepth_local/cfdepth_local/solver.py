from dataclasses import asdict
import numpy as np
from .contracts import ComponentState,SolverResult
from .coordinate import coordinate_minimum
from .energy import diagnostic_loop,model_args
from .state import mu_for,advance_component

def sweep_loop(S,order,starts,neighbors,weights,rows,degree,lower,upper,mid,beta,terrain,hard,lambda_b,lambda_t,mu,minimum):
    for color in range(4):
        for ii in range(starts[color],starts[color+1]):
            i=order[ii];ns=0.
            for k in range(8):
                j=neighbors[i,k]
                if j>=0:ns+=weights[rows[i],k]*S[j]
            soft=0. if hard[i] else lambda_t
            S[i]=minimum(degree[i],ns,lambda_b*beta[i],lower[i],upper[i],soft,terrain[i],hard[i],mu*beta[i],mid[i])

def final_audit(state,d,p):
    accepted=state.status in (3,4)
    budget=max(p.budgetRelative*state.base_primary,p.budgetAbsolute)
    passed=accepted and d['bad']==0 and d['hard']<=p.hardTolerance and d['residual']<=p.residualTolerance and d['primary']-state.base_primary<=budget
    return dict(passed=bool(passed),accepted=accepted,mu=mu_for(state,p),budget=budget,**d)

def solve_component(model,p,backend='python_reference',resume=None,callback=None):
    if backend=='python_reference':sweep,diag,minimum=sweep_loop,diagnostic_loop,coordinate_minimum
    else:
        from .kernels_numba import solver_kernels
        sweep,diag,minimum=solver_kernels(backend)
    args=model_args(model,p)
    def check(s,mu):
        vals=diag(s,*args,mu,minimum)
        return dict(zip(('J0','Jm','Q','primary','mid','total','residual','hard','bad'),map(float,vals)))
    order=np.argsort(model.topology.color,kind='stable').astype(np.int32)
    starts=np.concatenate(([0],np.cumsum(np.bincount(model.topology.color,minlength=4)))).astype(np.int64)
    if resume is None:
        S=model.boundary.initial_S.copy();base=S.copy();history=[];total=0
        d=check(S,0.)
        state=ComponentState(status=1 if model.boundary.eligible else 0,prev_primary=d['primary'],prev_total=d['total'])
    else:
        S=resume['S'].copy();base=resume['baseS'].copy();history=list(resume['history']);total=resume['total_sweeps'];state=ComponentState(**resume['state'])
    while state.status in (1,2):
        mu=mu_for(state,p)
        for _ in range(p.sweepsPerStage):sweep(S,order,starts,*args,mu,minimum)
        d=check(S,mu);total+=p.sweepsPerStage
        nxt,actions=advance_component(state,d,p)
        history.append(dict(state=asdict(state),diagnostics=d,next_status=nxt.status,actions=actions))
        if actions['saveBase']:base=S.copy()
        if actions['restoreBase']:S[:]=base
        state=nxt
        if callback is not None:callback(dict(S=S,baseS=base,state=asdict(state),history=history,total_sweeps=total),actions)
    audit=final_audit(state,check(S,mu_for(state,p)),p)
    if state.status in (3,4) and not audit['passed']:raise RuntimeError('Final component audit rejected')
    reason={0:'insufficient_boundary_support',3:'secondary_accepted',4:'primary_fallback',5:'primary_failed'}[state.status]
    if state.status==5 and history:
        d=history[-1]['diagnostics'];prev=history[-1]['state']
        if d['bad']:reason='primary_nonfinite_or_range_failure'
        elif d['hard']>p.hardTolerance:reason='primary_hard_constraint_failure'
        elif d['total']>prev['prev_total']+p.monotonicTolerance*max(1.,abs(prev['prev_total'])):reason='primary_nonmonotonic_failure'
        else:reason='primary_iteration_budget_exhausted'
    return SolverResult(S,base,state,history,audit,total,reason)
