from dataclasses import replace

def mu_for(state,p):
    return p.lambdaB*p.muRatios[state.attempt] if state.status in (2,3) else 0.

def advance_component(s,r,p):
    if s.status not in (1,2):return replace(s),dict(saveBase=False,restoreBase=False)
    feasible=r['bad']==0 and r['hard']<=p.hardTolerance
    healthy=feasible and r['total']<=s.prev_total+p.monotonicTolerance*max(1.,abs(s.prev_total))
    rel=lambda x,y:abs(x-y)/max(abs(y),p.objectiveFloor)
    steady=rel(r['primary'],s.prev_primary)<=p.objectiveTolerance and rel(r['total'],s.prev_total)<=p.objectiveTolerance
    stable=s.stable+1 if steady and healthy else 0
    sweeps=s.sweeps+p.sweepsPerStage
    converged=healthy and r['residual']<=p.residualTolerance and stable>=2
    exhausted=sweeps>=p.maxSweeps
    ready=s.status==1 and converged
    failed=s.status==1 and not converged and (exhausted or not healthy)
    budget=max(p.budgetRelative*s.base_primary,p.budgetAbsolute)
    accepted=s.status==2 and converged and r['primary']-s.base_primary<=budget
    rejected=s.status==2 and not accepted and (converged or exhausted or not healthy)
    retry=rejected and s.attempt+1<len(p.muRatios)
    fallback=rejected and not retry
    reset=ready or retry
    nxt=replace(s,status=2 if ready else 5 if failed else 3 if accepted else 4 if fallback else s.status,
                attempt=0 if ready else s.attempt+1 if retry else s.attempt,
                sweeps=0 if reset else sweeps,stable=0 if reset else stable,
                base_primary=r['primary'] if ready else s.base_primary,
                base_mid=r['mid'] if ready else s.base_mid)
    nxt.prev_primary=nxt.base_primary if reset else r['primary']
    nxt.prev_total=nxt.base_primary+p.lambdaB*p.muRatios[nxt.attempt]*nxt.base_mid if reset else r['total']
    return nxt,dict(saveBase=ready,restoreBase=retry or fallback)
