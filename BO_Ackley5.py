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
from jax.numpy.linalg import cholesky, solve, norm
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, ExpSineSquared, Matern,DotProduct

# from scipy.stats import norm#
from scipy.special import softmax
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

from joblib import Parallel, delayed
from functools import partial
from scipy.integrate import odeint, solve_ivp

import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid
from scipy.spatial.distance import pdist

from BO_via_PGF import *
# from JAX_SVGD_Langevin import *

from jax.config import config; config.update("jax_enable_x64", True)


#function to minimize

def Ackley5(X,d = 5):

    return( -20*np.exp(-.2*np.sqrt(1./d*(np.sum(X**2, axis = 1))))-np.exp((1./d*(np.sum(np.cos(2*np.pi*X), axis = 1))))+np.exp(1)+20)
true_min = 0
############# Minimise function ###########

# q = 3
# N = 10
# Num_Runs = 5
# Iterations = 15
# Start_Points = 50
# K = .5
# step_size = .5
# T = 2000
# R = 1
# M = 1000
# a = .01



q = 3
N = 10
Num_Runs = 5
Iterations = 15
Start_Points = 50
K = .5
step_size = .5
T = 3000
R = 1
M = 1000
a = .01


q_wass = 3
step_size_wass = .5
T_wass = 1500
a_wass = .005
method = 'wasserstein'
##Normal experiments
# Q = 10
# N = 15
# Num_Runs = 5
# Iterations = 20
# Start_Points = 10
# K = .1
# thresh = .015
# T = 2000
# R = 10

if __name__ == "__main__":
    #pool = Pool(2)
    if(method == 'wasserstein'):
        a = a_wass
        q = q_wass
        T = T_wass
        step_size = step_size_wass

    numpy.random.seed(1332)
    grid = Space([(-3., 3.), (-3., 3.),(-3., 3.),(-3., 3.),(-3., 3.)])

    lhs = Lhs(criterion="maximin", iterations=10000)
    obs_set_init = None #np.array(lhs.generate(grid.dimensions, Start_Points))
    obs_set_value_init = None #'Ackley(np.array(obs_set_init))

    
    numpy.random.seed(1234)
    seeds = numpy.random.randint(0,1000, 10)
    
    regret_Ackley_1_JAX , trace_Ackley_1_JAX , obs_set_Ackley_1, obs_set_value_Ackley_1 = BO_SVGD_vect(Ackley5,N = N , a = a, M = M , kernel_GP = Matern52_matrix, iterations = Iterations,q = q,num_runs = Num_Runs,T = T, step_size = step_size,R = R, grid = grid, k_stein = K, set_init = obs_set_init, set_init_val = obs_set_value_init,r_init = True, num_init = Start_Points, method = method, noise = False, seeds =seeds )
    
    



    

    
