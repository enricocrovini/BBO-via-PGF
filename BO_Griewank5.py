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
from jax.config import config; config.update("jax_enable_x64", True)

#function to minimize

def Griewank5(x, dim = 5):
    x = 100.*x
    """Griewank's function multimodal, symmetric, inseparable """
    partA = 0
    partB = 1
    for i in range(dim):
        partA += x[:,i]**2
        partB *= np.cos(x[:,i] / np.sqrt(i+1))
    return 1 + (partA/4000.0) - partB  
############# Minimise function ###########

q = 3
N = 10
Num_Runs = 5
Iterations = 20
Start_Points = 50
R = 1
M = 1000

# K = .5
# step_size = 1
# T = 3000
# R = 1
# M = 1000
# a = .0001

K = .5
step_size = .5
T = 5000
a = 0.0001 #.00001


step_size_wass = .5
T_wass = 3000
a_wass = .0005


method = 'stein'

if __name__ == "__main__":
    #pool = Pool(2)
    if(method == 'wasserstein'):
        a = a_wass
        T = T_wass
        step_size = step_size_wass
    

    grid = Space([(-5., 5.), (-5., 5.),(-5., 5.),(-5., 5.),(-5., 5.)])

    lhs = Lhs(criterion="maximin", iterations=10000)
    obs_set_init = None #np.array(lhs.generate(grid.dimensions, Start_Points))
    obs_set_value_init = None #'Ackley(np.array(obs_set_init))

    
    numpy.random.seed(1234)
    seeds = numpy.random.randint(0,1000, 10)
    #9:10 new a_wass = 0.001 T = 2500 too much regularisation
    #other = 0.0001 and T = 2000 results seem ok but not great
    #new try reg = 0.0005 T = 3000 with seeds from 5 to 9
    
    regret_Griewank5_JAX , obs_set_Griewank5, obs_set_value_Griewank5 = BO_SVGD_vect(Griewank5,N = N, M = M, a = a, kernel_GP = Matern52_matrix, iterations = Iterations,q = q,num_runs = Num_Runs,T = T, step_size = step_size,R = R, grid = grid, k_stein = K, set_init = obs_set_init, set_init_val = obs_set_value_init,r_init = True, num_init = Start_Points, method = method, seeds = seeds, noise = True)
    
    

    

    
