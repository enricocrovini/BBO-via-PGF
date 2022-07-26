import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']='1' 
os.environ['OPENBLAS_NUM_THREADS']='1'

os.environ["NUM_INTER_THREADS"]="1"
os.environ["NUM_INTRA_THREADS"]="1"

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
import jax.numpy as np
import numpy

from numpy.linalg import cholesky, solve
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, ExpSineSquared, Matern,DotProduct

from scipy.stats import norm
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

from joblib import Parallel, delayed
from functools import partial
from scipy.integrate import odeint, solve_ivp

from jax.config import config
from BO_via_PGF import *

#function to minimize

# from tqdm import tqdm 

import jax.numpy as np
from jax.ops import index, index_add, index_update
from jax import grad, jit, vmap, jacfwd, jacrev, partial
from jax import random
from jax import device_put
from jax.lax import fori_loop, scan, cond
from jax.tree_util import tree_map

import jax.numpy.linalg as lin
#from scipy.integrate import odeint
from jax.experimental.ode import odeint
from jax.experimental import host_callback

key_LL = random.PRNGKey(0)
tinit = 0
tend = int(1e1)
nt = int(50) #make it smaller to become rougher 
# nt = 20 #make it smaller to become rougher 
dt = tend/nt
s = 10.
b = 28.
r = 8./3.
burnin = 1000
times = np.linspace(burnin*dt, burnin*dt+tend, nt)
var = np.array([b,r])
var_truth = np.array([b,r])
y0 = np.array([0., 1., 1.05])

Sig0 = np.array([[0.98, 0.], [0., 0.78]]) #inverse variance of prior
Mu0 = np.array([27.11, 3.32])

num_par = 1000
sub_samp = 500
sigma = np.sqrt(0.05)


#function to minimize
# def progress_bar_factory(num_samples):
#     """Factory tat builds a progress bar decorator along
#     with the `set_tqdm_description` and `close_tqdm` f.66088115unctions
#     """

#     tqdm_pbar = tqdm(range(num_samples))

#     def set_tqdm_description(message):
#         tqdm_pbar.set_description(
#             message,
#             refresh=False,
#         )

#     def close_tqdm():
#         tqdm_pbar.close()

#     def _update_tqdm(arg, transform):
#         tqdm_pbar.update(arg)

#     @jit
#     def _progress_bar(arg, result):
#         """Updates tqdm progress bar of a scan/loop only if the iteration number is a multiple of the print_rate
#         Usage: carry = progress_bar((iter_num, print_rate), carry)
#         """
#         iter_num, print_rate = arg

#         result = cond(
#             iter_num % print_rate == 0,
#             lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=result),
#             lambda _: result,
#             operand=None,
#         )
#         return result

#     def progress_bar_scan(func):
#         """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
#         Note that `body_fun` must be looping over a tuple who's first element is `np.arange(num_samples)`.
#         This means that `iter_num` is the current iteration number
#         """
#         print_rate = int(num_samples / 10)

#         def wrapper_progress_bar(carry, iter_num):
#             iter_num = _progress_bar((iter_num, print_rate), iter_num)
#             return func(carry, iter_num)

#         return wrapper_progress_bar

#     return progress_bar_scan, set_tqdm_description, close_tqdm


def generate_initial_conditions(subkeys):    
    X0 = np.array([random.uniform(subkeys[0], shape=(num_par,), minval=27,maxval=29), random.uniform(subkeys[1], shape=(num_par,), minval=2.25, maxval=3.5) ])
    X0 = X0.T
    
    return X0

key_LL, *subkeys_LL = random.split(key_LL, 3)
X0 = generate_initial_conditions(subkeys_LL)

@jit
def ode_fn(y,t,params):
    b,r = params
    z1 = s *(y[1] - y[0])
    z2 = b*y[0] - y[1] - y[0]*y[2]
    z3 = -r*y[2] + y[0]*y[1]
    return np.stack([z1,z2,z3])
@jit
def calc_mom(y, dt):
    t0 = 0
    tau = 100
    tframe = np.arange(t0, t0+tau)
    x1 = y[t0:(t0+tau),0] 
    x2 = y[t0:(t0+tau),1] 
    x3 = y[t0:(t0+tau),2] 
    return np.mean(np.array([x1, x2, x3, x1**2, x2**2, x3**2, x1*x2, x1*x3, x2*x3]), axis=1)

@jit
def forward_solver(b,r, y0, time):
    var = np.array([b,r])
    sol = odeint(ode_fn, y0,time, var)

    #sol = odeint(ode_fn, y0, time, var)
    mom = calc_mom(sol,  dt)
    return sol, mom
def generate_random_starting_points_on_manifold(N):
    y0_list, _ = forward_solver(var_truth[0], var_truth[1], y0, np.linspace(burnin*dt, (burnin+10000)*dt, N))
    return y0_list

#############TRUTH###############
truth, G_truth = forward_solver(var_truth[0], var_truth[1], y0, times)
y = G_truth.flatten()

def estimate_Gamma(var, key, T, dt):
    
    key, sk1 = random.split(key)
    
    nsamples = 36
    mom_samples = np.zeros(shape=(nsamples,9))
    burnin = int(100/dt)
    timespan = np.arange(burnin * dt, burnin*dt+T, dt)
    b = var[0]
    r = var[1]

    init_choices = random.choice(sk1, np.arange(NUM_RANDOM_INIT_CONDITIONS), (num_par,), replace=True)
    y0_list = y0s[init_choices,:]

    for i in np.arange(nsamples):
        resi, momi = forward_solver(b ,r, y0_list[i,:], timespan)
        mom_samples = index_update(mom_samples, index[i,:], momi)
        #mom_samples[i,:] = momi
  
    G_out = np.cov(mom_samples.T)
    return mom_samples, G_out, key


@partial(jit, static_argnums=(2))
def log_prob_product(x, rng_key, prior=True):
    def body_fun1(z, j):
        _, GT_params = forward_solver(x[j,0], x[j, 1], y0_list[j,:], times)
        z = index_update(z, index[j, :],  (G_truth - GT_params))
        return z, None
    
    init_choices = random.choice(rng_key, np.arange(NUM_RANDOM_INIT_CONDITIONS), (num_par,), replace=True)
    y0_list = y0s[init_choices,:]
     
    z, _ = scan(body_fun1, np.zeros(shape=(x.shape[0],9)), np.arange(x.shape[0]))
    
    z1 = np.dot(z, Gamma_out)
    r1 = np.sum(z1*z, axis=1)
    
    if prior:
        x1 = np.dot(x-Mu0, Sig0)
        r2 = np.sum(x1*(x-Mu0), axis=1)
        return -0.5*(r1 + r2)
    else:
        return -0.5*r1

@partial(jit, static_argnums=(2))
def log_prob(x, rng_key, prior=True):
    def body_fun1(z, j):
        _, GT_params = forward_solver(x[j,0], x[j, 1], y0_list[j,:], times)
        z = index_update(z, index[j, :],  (G_truth - GT_params))
        return z, None
    
    init_choices = random.choice(rng_key, np.arange(NUM_RANDOM_INIT_CONDITIONS), (num_par,), replace=True)
    y0_list = y0s[init_choices,:]
     
    z, _ = scan(body_fun1, np.zeros(shape=(num_par,9)), np.arange(num_par))
    
    
    z1 = np.dot(z, Gamma_out)
    r1 = np.sum(z1*z)
    
    if prior:
        x1 = np.dot(x-Mu0, Sig0)
        r2 = np.sum(x1*(x-Mu0), axis=1)
        return -0.5*(r1 + r2)
    else:
        return -0.5*(r1)   

@jit
def GX(x,key):
    def body_fun1(z, j):
        _, GT_params = forward_solver(x[j,0], x[j, 1], y0_list[j,:], times)
        z = index_update(z, index[j, :],  GT_params)
        return z, None
    
    init_choices = random.choice(key, np.arange(NUM_RANDOM_INIT_CONDITIONS), (num_par,), replace=True)
    y0_list = y0s[init_choices,:]
 
    z, _ = scan(body_fun1, np.zeros(shape=(x.shape[0],9)), np.arange(x.shape[0]))
    
    
    return z
def neg_log_prob(X, key):
    return -log_prob(X, key,False)/num_par

NUM_RANDOM_INIT_CONDITIONS = 100000
y0s = generate_random_starting_points_on_manifold(NUM_RANDOM_INIT_CONDITIONS)

def estimate_Gamma(var, key, T, dt):
    
    key, sk1 = random.split(key)
    
    nsamples = 36
    mom_samples = np.zeros(shape=(nsamples,9))
    burnin = int(100/dt)
    timespan = np.arange(burnin * dt, burnin*dt+T, dt)
    b = var[0]
    r = var[1]

    init_choices = random.choice(sk1, np.arange(NUM_RANDOM_INIT_CONDITIONS), (num_par,), replace=True)
    y0_list = y0s[init_choices,:]

    for i in np.arange(nsamples):
        resi, momi = forward_solver(b ,r, y0_list[i,:], timespan)
        mom_samples = index_update(mom_samples, index[i,:], momi)
        #mom_samples[i,:] = momi
  
    G_out = np.cov(mom_samples.T)
    return mom_samples, G_out, key

mom_sam, G_out,key_LL = estimate_Gamma(var_truth, key_LL,tend, dt)
Gamma_out = np.linalg.inv(G_out)

# np.savez('params_lorentz63.npz', X0 = X0, y = y, var=var_truth, npar=num_par, time=times, Gout=Gamma_out)


def LL_new(X):
    config.update("jax_enable_x64", False)
#     a = ((-numpy.array(log_prob_product(numpy.abs(numpy.array(x), dtype=numpy.float), subkeys_LL, False), dtype=numpy.float))**.3)
    a = [((-numpy.array(log_prob(numpy.abs(numpy.array(x[np.newaxis],dtype = np.float32), dtype=numpy.float), subkeys_LL, False), dtype=numpy.float))**.3)for x in X]
    config.update("jax_enable_x64", True)
    return(np.array(a).squeeze())

N = 10
Num_Runs = 5
Iterations = 20
Start_Points = 5


K = .5
step_size_stein = .5
T_stein = 4000
R = 1
M = 1000
a_stein = .001 
q = 2


q_wasserstein = 2
step_size_wasserstein = .1
a_wasserstein = .01 
T_wasserstein = 2000 #




method = 'wasserstein'

if __name__ == "__main__":
    print('\n\n', method, '\n\n')
    
    if(method =='stein'):
        a = a_stein
        T = T_stein
        step_size = step_size_stein
#         noise = False
        noise = True

    elif(method == 'wasserstein'):
        a = a_wasserstein
        q = q_wasserstein
        T = T_wasserstein
        step_size = step_size_wasserstein
        noise = True

    #pool = Pool(2)
    

#     numpy.random.seed(1332)
    grid = Space([(20., 40.), (0., 10.)])

    lhs = Lhs(criterion="maximin", iterations=10000)
    obs_set_init = None #np.array(lhs.generate(grid.dimensions, Start_Points))
    obs_set_value_init = None #'Ackley(np.array(obs_set_init))

    
    numpy.random.seed(1234)
    seeds = numpy.random.randint(0,1000, 10)


    regret_LL_1 , trace_LL_1 , obs_set_LL_1, obs_set_value_LL_1  = BO_SVGD_vect(LL_new, M=M,N=N, a = a, kernel_GP = Matern52_matrix,norm_data = True, iterations = Iterations,q = q,num_runs = Num_Runs,T = T, step_size = step_size,R = R, grid = grid, k_stein = K, set_init = obs_set_init, set_init_val = obs_set_value_init,r_init = True,  num_init = Start_Points, method = method, seeds = seeds, noise = noise)
    


    
    


    

    
