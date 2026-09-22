from .contracts import Prepared
from .domains import build_domains
from .topology import build_topology
from .boundary import boundary_arrays,component_boundary

def prepare_small(inputs,p):
    domain=build_domains(inputs);first,beta=boundary_arrays(inputs,domain,p);result=[]
    for c in domain.component_table:
        t=build_topology(domain.component_id==c['id'],inputs.grid,p.lambdaC)
        z=inputs.dem[t.rows,t.cols];hard=domain.hard[t.rows,t.cols]
        b=component_boundary(first,beta,t,z,hard,p)
        result.append(Prepared(t,z,hard,b,c['id']))
    return result
