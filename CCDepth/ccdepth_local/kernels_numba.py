"""Compiled versions of the reference loops; strict IEEE arithmetic."""
from numba import njit,prange,set_num_threads
from .coordinate import coordinate_minimum
from .energy import diagnostic_loop
from .solver import sweep_loop
from .boundary import first_pass,peer_pass
from .products import gradient_loop

minimum_kernel=njit(cache=True,fastmath=False)(coordinate_minimum)
diagnostic_kernel=njit(cache=True,fastmath=False)(diagnostic_loop)
sweep_serial=njit(cache=True,fastmath=False)(sweep_loop)
first_kernel=njit(cache=True,fastmath=False)(first_pass)
peer_kernel=njit(cache=True,fastmath=False)(peer_pass)
gradient_kernel=njit(cache=True,fastmath=False)(gradient_loop)

@njit(cache=True,fastmath=False,parallel=True)
def sweep_parallel(S,order,starts,neighbors,weights,rows,degree,lower,upper,mid,beta,terrain,hard,lambda_b,lambda_t,mu,minimum):
    for color in range(4):
        for ii in prange(starts[color],starts[color+1]):
            i=order[ii];ns=0.
            for k in range(8):
                j=neighbors[i,k]
                if j>=0:ns+=weights[rows[i],k]*S[j]
            soft=0. if hard[i] else lambda_t
            S[i]=minimum(degree[i],ns,lambda_b*beta[i],lower[i],upper[i],soft,terrain[i],hard[i],mu*beta[i],mid[i])

def solver_kernels(backend):
    if backend not in ('numba_serial','numba_parallel'):raise ValueError(backend)
    return (sweep_parallel if backend=='numba_parallel' else sweep_serial),diagnostic_kernel,minimum_kernel

def boundary_kernels():return first_kernel,peer_kernel

def configure_threads(n):set_num_threads(n)
