from __future__ import division
import itertools
import numpy
import jax.numpy as np
from jax.numpy.linalg import cholesky, solve, svd
import numpy as onp
import jax
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from jax import random 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, ExpSineSquared, Matern,DotProduct
from sklearn.metrics import pairwise_distances
from jax import lax

# from jax.scipy.integrate import solve_ivp
from jax.nn import softmax
from jax.scipy.stats import norm

from joblib import Parallel, delayed
from functools import partial
from scipy.integrate import odeint, solve_ivp

from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs 
from jax import pmap, vmap


#######

##### Define GP utilities ####
@jax.jit
def RBF_matrix(X,Y,gamma):
    d =np.linalg.norm((X[...,:, None,:] - Y[...,None,:, :]), axis = -1)
    return np.exp(-d/(2.0*gamma**2))

#grad with respect to 1st component
@jax.jit
def gradRBF(X, Y,  gamma):
    K1 = RBF_matrix(X, Y, gamma)
    z = -(np.multiply((X[:, np.newaxis, :]- Y[np.newaxis, :, :]), K1[:,:, np.newaxis]))/gamma**2
    return np.transpose(z, axes=[0,2,1])

@jax.jit
def gradgradRBF(X, Y,  gamma):
    
    dist = np.transpose(X[:, np.newaxis, :]- Y[np.newaxis, :, :],axes = [0,2,1])[:,:,np.newaxis,:]
    dist_t = np.transpose(X[:, np.newaxis, :]- Y[np.newaxis, :, :], axes = [0,2,1] )[:,np.newaxis,:,:]
    product = np.multiply(dist, dist_t)
    dim =  X.shape[1]
    identity = np.eye(dim)[np.newaxis,:,:,np.newaxis]
    add_term = np.repeat(identity, X.shape[0], axis = 0)
    add_term = np.repeat(add_term, Y.shape[0], axis = -1)*(gamma**2)
    
    const = RBF_matrix(X, Y, gamma)[:,np.newaxis, np.newaxis,:]/gamma**4
    
    cov_derivative = np.multiply(const,(add_term - product))
    
    return(cov_derivative)

def p_Kernel(X,Y,p,gamma):
    return np.exp(-pairwise_distances(X, Y)**p/(gamma**p))

def grad_p_Kernel(X,Y,p,gamma):
    
    K = -p*p_Kernel(X,Y,p,gamma)/gamma**p
    z = (np.multiply((X[:, np.newaxis, :]- Y[np.newaxis, :, :]), K[:,:, np.newaxis]))
    Gr = np.multiply(z, pairwise_distances(X, Y)[:,:,np.newaxis])
    
    return(np.transpose(Gr, axes=[0,2,1]))

# @jax.jit
# def Matern52_matrix(X,Y,l):
#     d =np.linalg.norm((X[:, None,:] - Y[None,:, :]), axis = -1)

#     return(1+np.sqrt(5)*d/l + 5*d**2/(3*l**2))*np.exp(-np.sqrt(5)*d/l)
    

def Matern52_matrix(X,Y,l):
    d =np.linalg.norm((X[...,:, None,:] - Y[...,None,:, :]), axis = -1)

    return(1+np.sqrt(5)*d/l + 5*d**2/(3*l**2))*np.exp(-np.sqrt(5)*d/l)    

#grad with respect to 1st component

@jax.jit
def gradMatern52(X,Y,l):
    d = np.linalg.norm((X[:, None,:] - Y[None,:, :]), axis = -1)
    
    K = ((5)/(3*l**2) + 5*np.sqrt(5)*d/(3*l**3))*np.exp(-np.sqrt(5)*d/l)
    deriv = (X[:, np.newaxis, :]- Y[np.newaxis, :, :])
    
    dim = X.shape[1]

    derivative = np.transpose(-np.multiply(deriv,K[:,:, np.newaxis]), axes=[0,2,1])
    
    return(derivative)

@jax.jit
def gradgradMatern52(X,Y,l):
    d = np.linalg.norm((X[:, None,:] - Y[None,:, :]), axis = -1)
    
    dist = np.transpose(X[:, np.newaxis, :]- Y[np.newaxis, :, :],axes = [0,2,1])[:,:,np.newaxis,:]
    dist_t = np.transpose(X[:, np.newaxis, :]- Y[np.newaxis, :, :], axes = [0,2,1] )[:,np.newaxis,:,:]
    
    product = np.multiply(dist, dist_t)*(-5/l**2)
    const = (np.exp(-np.sqrt(5)*d/l)*5/(3*l**2))[:,np.newaxis, np.newaxis,:]
    
    dim = X.shape[1]

    identity = np.eye(dim)[np.newaxis,:,:,np.newaxis]
    add_term = np.repeat(identity, X.shape[0], axis = 0)
    add_term = np.repeat(add_term, Y.shape[0], axis = -1)*((1+np.sqrt(5)*d/l)[:,np.newaxis, np.newaxis,:])
    
    cov_derivative = np.multiply(const,(add_term + product))
        
    return(cov_derivative)

    
#grad with respect to 1st component


# @jax.partial(jax.jit, static_argnums=(3,6,7,8,9))
# def return_grad(X_obs,y_obs, Xtest, kernel, params_list, err = 1e-6,return_joint = True, start_points = 10, iteration = 20, q = 10):
#     X = X_obs[0:(start_points + q*iteration)]
#     y = y_obs[0:(start_points + q*iteration)]
    
#     if(kernel.__name__ == 'RBF_matrix'):
#         kernel = RBF_matrix 
#         grad = gradRBF
#         gradgrad = gradgradRBF
#     elif(kernel.__name__ == 'Matern52_matrix'):
#         kernel = Matern52_matrix
#         grad = gradMatern52
#         gradgrad = gradgradMatern52
            
# #     kernel = Matern52_matrix
# #     grad = gradMatern52
# #     gradgrad = gradgradMatern52

#     params = params_list[0]
#     noise_kernel = params_list[1]
    
#     N_obs, n_star = len(X), len(Xtest)
#     mean = np.mean(y)
#     K = kernel(X, X, params)+ np.eye(N_obs)*noise_kernel
#     L = cholesky(K + err*np.eye(N_obs))
#     alpha = solve(L.T,solve(L, y- mean))
        
        
#     if(return_joint):
#         #return joint process and its gradient
        
#         dim = X.shape[1]

#         kernel_val = kernel(Xtest, X, params)
#         kernel_grad  = grad(Xtest, X,  params)

#         kernel_grad_joint = np.append(kernel_val[:,np.newaxis,:], kernel_grad, axis = 1)


#         exp_grad_zero_mean = kernel_grad_joint @ alpha 
        
#         mean_reshape = np.repeat(np.append(np.array(mean.squeeze()),np.zeros(dim))[None,:] , (exp_grad_zero_mean.shape[0]), axis = 0)
        
        
#         exp_grad = np.expand_dims(mean_reshape, -1) + np.expand_dims(exp_grad_zero_mean, -1)


#         grad_star, cov_star = grad(Xtest, Xtest, params), gradgrad(Xtest, Xtest, params)
 
#         kernel_val_star = kernel(Xtest, Xtest, params) + np.eye(n_star)*noise_kernel


#         A = np.append(kernel_val_star[:,np.newaxis, np.newaxis, :], - grad_star[:, np.newaxis, :, :] , axis= 2)
    
#         B = np.append(- np.transpose(grad_star, (2,1,0))[:,:,np.newaxis, :], cov_star , axis= 2)
        
#         cov_joint = np.append(A, B, axis = 1)
        

#         covariance_grad = cov_joint - np.einsum('ijk,kl,lmp->ijmp', kernel_grad_joint, solve(K+np.eye(N_obs)*err,np.eye(N_obs)), kernel_grad_joint.T)

#         return(exp_grad, covariance_grad)

#     #return gradient sampled independent of process
    
#     kernel_val = kernel(Xtest, X, params)
#     kernel_grad  = grad(Xtest, X, params)


#     exp_grad = kernel_grad @ alpha       
#     exp_grad_start, cov_star = grad(Xtest, Xtest, params), gradgrad(Xtest, Xtest, params)
#     covariance_grad = cov_star - np.einsum('ijk,kl,lmp->ijmp',kernel_grad, solve(K+np.eye(X.shape[0])*err,np.eye(X.shape[0])),kernel_grad.T)

#     return(exp_grad, covariance_grad)


#################################################################################################







############# Implementation of BO #################



    







#compute qEI using MCMC to compare the output of different chains
def qEI_MCMC(X,gpr, curr_min):
    M = gpr.sample_y(X, 7000)
    sims_full = curr_min - np.min(M, axis = 0)
    sims = np.clip(sims_full, 0)
    qI_sim = np.mean(sims)         
    return qI_sim




# @jax.jit
def generate_new_sample(grid, q):
    lhs = Lhs(criterion="maximin", iterations=100)
    new_start = np.array(lhs.generate(grid.dimensions, q))
    return(new_start)




def log_regret(X, obs_val):
    return(np.log(np.min(obs_val)))


def distance_from_min(x):
    true_min = np.zeros((1, x.shape[1]))
    return(np.min(np.linalg.norm(x - true_min, axis = 1)))


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

@partial(jax.jit, static_argnums=(3,6,7,8))
def return_grad_optimised(X_obs,y_obs, Xtest, kernel, params_list, err, start_points, iteration, q):
    
    n_star = len(Xtest)    

    X = X_obs[0:(start_points + n_star*iteration)]
    y = y_obs[0:(start_points + n_star*iteration),None]
    N_obs = len(X)

#     err = 1e-6    
    params = params_list[0]
    noise_kernel = params_list[1]
    
        
    kernel = Matern52_matrix
    grad = gradMatern52
    gradgrad = gradgradMatern52
    
#     kernel = RBF_matrix
#     grad = gradRBF
#     gradgrad = gradgradRBF
    
    n_star = len(Xtest)    


    N_obs = len(X)
    err = 1e-10
    
    
    mean = np.mean(y)
    K = kernel(X, X, params) + np.eye(N_obs)*noise_kernel
    L = cholesky(K + err*np.eye(N_obs))
    alpha = solve(L.T,solve(L, y- mean))
    K_inverse = solve(K+np.eye(N_obs)*err,np.eye(N_obs))
           
        
    dim = X.shape[1]

    kernel_val = kernel(Xtest, X, params)
    kernel_grad  = grad(Xtest, X,  params)

    #kernel_grad_joint = np.append(kernel_val[:,np.newaxis,:], kernel_grad, axis = 1).reshape(n_star*(dim+1), N_obs)
    kernel_grad_reshape = kernel_grad.reshape(n_star*(dim), N_obs)
    kernel_grad_joint = np.block([[kernel_val], [kernel_grad_reshape]])
    
#     kernel_grad_t_reshape = np.transpose(kernel_grad, (2,1,0)).reshape(n_star*(dim), N_obs).T
    kernel_grad_t_joint = np.block([kernel_val.T, kernel_grad_reshape.T])

    exp_grad_zero_mean = kernel_grad_joint @ alpha 
        
    mean_extended = np.append(np.ones(n_star)*mean.flatten(), np.zeros(n_star*dim)).flatten()#np.repeat(np.append(np.array(mean.squeeze()),np.zeros(dim))[None,:] , (exp_grad_zero_mean.shape[0]), axis = 0)
        
        
#     exp_grad = np.expand_dims(mean_reshape, -1) + np.expand_dims(exp_grad_zero_mean, -1)

    exp_grad = exp_grad_zero_mean + mean_extended[:,None]
    

    grad_star, cov_star = grad(Xtest, Xtest, params), gradgrad(Xtest, Xtest, params)

    kernel_val_star = kernel(Xtest, Xtest, params) + np.eye(n_star)*noise_kernel
 

    grad_star_reshape = grad_star.reshape(n_star*(dim), n_star) #np.append(kernel_val_star[:,np.newaxis,:], grad_star, axis = 1).reshape(n_star*(dim+1), n_star)
    grad_star_t_reshape = grad_star_reshape.T #np.transpose(grad_star, (2,1,0)).reshape(n_star*(dim), n_star).T #np.append(kernel_val_star[:,np.newaxis,:], grad_star, axis = 1).reshape(n_star*(dim+1), n_star)

    
    cov_star_reshape = cov_star.reshape(1,n_star*(dim),cov_star.shape[2], cov_star.shape[3]).swapaxes(-1,-2).reshape(1,n_star*(dim),n_star*(dim),1).squeeze()    

#     A = np.append(kernel_val_star[:,np.newaxis, np.newaxis, :],  grad_star[:, np.newaxis, :, :] , axis= 2)
    
#     B = np.append(np.transpose(grad_star, (2,1,0))[:,:,np.newaxis, :], cov_star , axis= 2)
        
    cov_joint = np.block([[kernel_val_star, grad_star_t_reshape], [grad_star_reshape, cov_star_reshape]])
#     print(cov_joint.shape)
#     print(kernel_grad_joint.shape)

#     covariance_grad = cov_joint - np.einsum('ijk,kl,lmp->ijmp', kernel_grad_joint, K_inverse, kernel_grad_joint.T)

#     return(exp_grad, covariance_grad)


    
    covariance_grad = cov_joint - kernel_grad_joint @  K_inverse @ kernel_grad_t_joint

    return(exp_grad, covariance_grad)




@partial(jax.jit, static_argnums=(2, 6, 8, 9, 10,11,12,13))
def compute_chains_compiled_opt(obs_set, obs_set_value,kernel_GP,kernel_params,step_size , k_stein , T ,a, M  , method, dim, q , start_points , iteration, current_point, key):

    if(kernel_GP.__name__ == 'RBF_matrix'):
        kernel = RBF_matrix 
    elif(kernel_GP.__name__ == 'Matern52_matrix'):
        kernel = Matern52_matrix

    points = []
    grad = []
    
    N = current_point.shape[0]
#     X = obs_set[0:(start_points + N*iteration)]
#     y = obs_set_value[0:(start_points + N*iteration)]
#     N_obs = len(X)
    err = 1e-8
#     mean = np.mean(y)
#     K = kernel(X, X, kernel_params)
#     L = cholesky(K + err*np.eye(N_obs))
#     alpha = solve(L.T,solve(L, y- mean))
#     K_inverse = solve(K+np.eye(N_obs)*err,np.eye(N_obs))
    current_point = current_point.squeeze()
    func_carryover = partial(run_chain_opt, obs_set , obs_set_value ,kernel  ,kernel_params  ,err, key, k_stein, dim, q, M, a, start_points, iteration, step_size, method)
    
#     current_point_list = np.tile(current_point, (T,1,1))
    current_point_list = np.tile(current_point, (T,1,1))

    result, final = lax.scan(func_carryover, current_point, current_point_list)
        
    return(result, final)



    


''' check current point and total points '''
@partial(jax.jit, static_argnums=(2,3,4))
def return_grad_stein_combinations(mu_joint,cov_joint ,  dim, q, M, curr_min , k_stein, current_point,a , key):
    N = current_point.shape[0]
 #     mu_joint = mu_grad_joint.flatten()
#     cov_joint = cov_grad_joint.reshape(1,N*(dim+1),cov_grad_joint.shape[2], cov_grad_joint.shape[3]).swapaxes(-1,-2).reshape(1,N*(1+dim),N*(dim+1),1).squeeze()    
    
#     process = random.normal(key, shape = (M,N*(dim+1)))  #random.multivariate_normal(key, np.zeros_like(mu_joint), np.eye(cov_joint.shape[0]), (M,)) 
    
#     U, S, _ = svd(cov_joint+ 1e-8*np.identity(cov_joint.shape[0]))
    
#     L = U * np.sqrt(S[..., None, :])
    
#     joint_process = (mu_joint + np.einsum('ij, kj -> ki', L, process)).reshape(M, N , dim+1)    

    process = random.normal(key, shape = (M,N*(dim+1)))  #random.multivariate_normal(key, np.zeros_like(mu_joint), np.eye(cov_joint.shape[0]), (M,)) 
    
    U, S, _ = svd(cov_joint+ 1e-9*np.identity(cov_joint.shape[0]))
    
    L = U * np.sqrt(S[..., None, :])
    
    joint_process_samples = (mu_joint.T + np.einsum('ij, kj -> ki', L, process))
    
    process_samples = joint_process_samples[:,0:N]
    grad_samples = joint_process_samples[:,N:]
    
    joint_process = np.append(process_samples[:,:,None], grad_samples.reshape(M,N,dim), axis = 2)
    
#     joint_process = np.transpose(joint_process_samples.reshape(M, dim+1, N)  , (0,2,1))
    
    l_combinations = np.array(list(itertools.combinations(np.arange(N), q)))
    joint_process_combinations = joint_process[:,l_combinations,: ] #Mxnum_combxqxdim+1
    
    combinations_current_points = current_point[l_combinations,:] # num_comb x q x dim +1
        
    process_real = joint_process_combinations[:,:,:,0] #Mxnum_combxq
    grad_process = joint_process_combinations[:,:,:,1:] #Mxnum_combxqxdim
    
    alpha_pos = (curr_min - process_real)
    
    #append a 0 ~ used to approximate the function max(0,)
    alpha_pos = np.append(alpha_pos, np.zeros((alpha_pos.shape[0],alpha_pos.shape[1], 1)) , axis = 2)#Mx num_comb x q+1

    #take softmax and discard the term related to 0
    EI_soft = jax.nn.softmax(alpha_pos , axis = 2)[:,:,:-1] # cancel the 0 term that was   #M x num_comb x q 
            
    #find expected gradient for all combinations      
    gradient =  -np.mean(np.multiply(EI_soft[:, :, :,np.newaxis], grad_process), axis = 0) # num_comb x q x dim
    
    #evaluate Kernel Stein for N location vs all combination 
    Kern_Stein = Matern52_matrix(current_point[np.newaxis,:,:], current_point[l_combinations,:], k_stein)  #num_comb x N x q
    
    #evaluate gradient_stein for all combinations
    grad_stein = np.einsum('ijk, ikl -> ijl', Kern_Stein , gradient).squeeze()  #num_comb x N x dim
    
    #Evaluate U statistics 
    U_stat = np.mean(grad_stein, axis = 0) # N x dim 
                    
    gradKern_Stein =  a*gradMatern52(current_point,current_point, k_stein)

    grad_complete = + U_stat + np.mean(gradKern_Stein, axis = 0).T
    
    return(grad_complete)


@partial(jax.jit, static_argnums=(2,3,4))
def return_grad_wasserstein_combinations(mu_joint,cov_joint ,  dim, q, M, curr_min , k_stein, current_point,a , key):
    N =current_point.shape[0]   
#     mu_joint = mu_grad_joint.flatten()
#     cov_joint = cov_grad_joint.reshape(1,N*(dim+1),cov_grad_joint.shape[2], cov_grad_joint.shape[3]).swapaxes(-1,-2).reshape(1,N*(1+dim),N*(dim+1),1).squeeze()    
    
#     process = random.normal(key, shape = (M,N*(dim+1)))  #random.multivariate_normal(key, np.zeros_like(mu_joint), np.eye(cov_joint.shape[0]), (M,)) 
    
#     U, S, _ = svd(cov_joint+ 1e-8*np.identity(cov_joint.shape[0]))
    
#     L = U * np.sqrt(S[..., None, :])
    
#     joint_process = (mu_joint + np.einsum('ij, kj -> ki', L, process)).reshape(M, N , dim+1)    

    process = random.normal(key, shape = (M,N*(dim+1)))  #random.multivariate_normal(key, np.zeros_like(mu_joint), np.eye(cov_joint.shape[0]), (M,)) 
    
    U, S, _ = svd(cov_joint+ 1e-9*np.identity(cov_joint.shape[0]))
    
    L = U * np.sqrt(S[..., None, :])
    joint_process_samples = (mu_joint.T + np.einsum('ij, kj -> ki', L, process))
    
    process_samples = joint_process_samples[:,0:N]
    grad_samples = joint_process_samples[:,N:]
    
    joint_process = np.append(process_samples[:,:,None], grad_samples.reshape(M,N,dim), axis = 2)
#     joint_process = np.transpose(joint_process_samples.reshape(M, dim+1, N)  , (0,2,1))
        
    process = joint_process[:,:,0] #MxN
    grads = joint_process[:,:,1:]  #MxNxdim
    
    l_combinations = np.array(list(itertools.combinations(np.arange(N), q-1)))
    n_comb = len(l_combinations)
    
    joint_process_combinations = joint_process[:,l_combinations,: ] #Mxnum_combx q-1 xdim+1
    combinations_current_points = current_point[l_combinations,:] # num_comb x q-1 x dim 
        
    process_real = joint_process_combinations[:,:,:,0] #Mxnum_combx q-1
    grad_process = joint_process_combinations[:,:,:,1:] #Mxnum_combx q-1 xdim
    
    
    
    process_real_reshape = np.transpose(np.tile(process_real[:,None], (1,N,1,1)), (1,0,2,3)) #N x M x ncomb x q-1
    process_reshape = np.transpose(np.tile(process[:,None], (n_comb,1))[:,:,:,None], (2,0,1,3))
    process_real_combinations = np.append(process_real_reshape,process_reshape, axis = -1 )
    
    
#     process_real_combinations = np.array([np.append(process_real,np.tile(process[:,i], (n_comb,1)).T[:,:,None], axis = -1) for i in range(N)]) #N x M x num_combx q
    
    grad_process_reshape = np.transpose(np.tile(grad_process[:,None], (1,N,1,1,1)), (1,0,2,3,4)) #NxMxnum_combx q-1 xdim
    grads_reshape = np.transpose(np.tile(grads[:,None], (n_comb,1,1))[:,:,:,None,:], (2,0,1,3,4)) #N x M x ncomb x 1 x dim
    process_grad_combinations = np.append(grad_process_reshape, grads_reshape, axis = -2)
    
#     process_grad_combinations = np.array([np.append(grad_process,np.transpose(np.tile(grads[:,i], (n_comb,1,1)), (1,0,2))[:,:,None], axis = -2) for i in range(N)]) #N x M x num_combx q x dim

    
    alpha_pos = (curr_min - process_real_combinations) # N x M x num_combx q
    
    #append a 0 ~ used to approximate the function max(0,)
    alpha_pos = np.append(alpha_pos, np.zeros((alpha_pos.shape[0],alpha_pos.shape[1],alpha_pos.shape[2], 1)) , axis = -1)# N x Mx num_comb x q+1

    #take softmax and discard the term related to 0
    EI_soft = jax.nn.softmax(alpha_pos , axis = -1)[:,:,:,:-1] # cancel the 0 term that was  # N xM x num_comb x q
            
    #find expected gradient for all combinations      
    gradient =  -np.mean(np.multiply(EI_soft[:, :, :, -1,np.newaxis], process_grad_combinations[...,-1,:]), axis = 1) # N x num_comb x 1 x dim
    
    #Evaluate U statistics 
    U_stat = np.mean(gradient, axis = 1).squeeze() # N x dim 
                    

    noise = random.normal(key, shape = (N, dim))
    
    penalisation = a*noise

    grad_complete = + U_stat + penalisation
    
    return(grad_complete)



# @jax.partial(jax.jit, static_argnums=(2,4,6,7,8,9))
# def qEI_MCMC_combinations(obs_set, obs_set_value, kernel_GP,kernel_params , M  , method, dim, q , start_points , iteration, current_point, key):
    
    
    
#     if(kernel_GP.__name__ == 'RBF_matrix'):
#         kernel = RBF_matrix 
#     elif(kernel_GP.__name__ == 'Matern52_matrix'):
#         kernel = Matern52_matrix

#     points = []
#     grad = []
    
#     X = obs_set[0:(start_points + q*iteration)]
#     y = obs_set_value[0:(start_points + q*iteration)]
    
#     curr_min = np.min(y)
    
#     N_obs = len(X)
#     err = 1e-6
#     mean = np.mean(y)
#     K = kernel(X, X, kernel_params)
#     L = cholesky(K + err*np.eye(N_obs))
#     alpha = solve(L.T,solve(L, y- mean))
#     K_inverse = solve(K+np.eye(N_obs)*err,np.eye(N_obs))
    
#     mu_grad_joint, cov_grad_joint =  return_grad_optimised(obs_set, obs_set_value, current_point, kernel,kernel_params, err, start_points, iteration, q, mean, K, L, alpha, K_inverse)
    
#     N = current_point.shape[0]
    
#     mu_joint = mu_grad_joint.flatten()
#     cov_joint = cov_grad_joint.reshape(1,N*(dim+1),cov_grad_joint.shape[2], cov_grad_joint.shape[3]).swapaxes(-1,-2).reshape(1,N*(1+dim),N*(dim+1),1).squeeze()    
    
#     process = random.normal(key, shape = (M,N*(dim+1)))  #random.multivariate_normal(key, np.zeros_like(mu_joint), np.eye(cov_joint.shape[0]), (M,)) 
    
#     U, S, _ = svd(cov_joint+ 1e-8*np.identity(cov_joint.shape[0]))
    
#     L = U * np.sqrt(S[..., None, :])
    
#     joint_process = (mu_joint + np.einsum('ij, kj -> ki', L, process)).reshape(M, N , dim+1)  
    
#     l_combinations = np.array(list(itertools.combinations(np.arange(N), q)))
#     joint_process_combinations = joint_process[:,l_combinations,: ] #Mxnum_combxqxdim+1
    
#     combinations_current_points = current_point[l_combinations,:] # num_comb x q x dim 
        
#     process_real = joint_process_combinations[:,:,:,0] #Mxnum_combxq
    
#     alpha_pos = (curr_min - np.min(process_real, axis = -1)) #Mxnum_comb

#     sims = np.clip(alpha_pos, 0)
#     qI_sim = np.mean(sims , axis = 0)    #num_comb     
#     return qI_sim


# qEI_MCMC_combinations_opt = jax.jit(vmap(qEI_MCMC_combinations, 
#                                 in_axes=(None , None,None,None,None,None,None,None,None,None,0,0)),
#                     static_argnums=(2,4,6,7,8,9))


@jax.partial(jax.jit, static_argnums=(2, 7, 8, 9,11, 12,14))
def run_chain_opt(obs_set, obs_set_value,kernel,kernel_params, err, key,  k_stein, dim, q, M, a, start_points, iteration,step_size, method, current_point, el ):
    N = current_point.shape[0]
    curr_min = np.min(obs_set_value[0:start_points + (iteration)*N ])
    
    
    
    key, subkey = random.split(key)
    
    
    mu_grad_joint, cov_grad_joint =  return_grad_optimised(obs_set, obs_set_value, current_point,kernel,kernel_params, err, start_points, iteration, q)
    

    grad_complete  = method(mu_grad_joint,cov_grad_joint, dim, q, M, curr_min , k_stein, current_point,a, subkey)

    
    current_point = current_point + step_size * grad_complete
    
    return current_point, (current_point, np.linalg.norm(grad_complete))







                    
def qEI_Stein_vect_opt(obs_set, obs_set_value,kernel, kernel_params, current_point , T, a, R, q, M,k_stein,step_size,start_points , iteration,  method):

    
    key = random.PRNGKey(0)

    key, subkeys = random.split(key, R+1)

    dim = obs_set.shape[-1]
    
    points, final = compute_chains_compiled_opt(obs_set, obs_set_value , kernel, kernel_params ,
                        step_size ,k_stein ,T,a, M, method, dim,q, start_points, iteration, current_point, subkeys)
    
    return(points, final)  





next_points_qEI_opt = jax.jit(vmap(qEI_Stein_vect_opt, 
                                in_axes=(0, 0,None,0,0,None,None,None,None,None,None,None,None,None, None)),
                    static_argnums=(2, 5, 7, 8,9, 10,12,13,14))

#the algorithm proposes N particles as batch

def BO_SVGD_vect(func, iterations, grid, set_init,set_init_val, kernel_GP = RBF_matrix, regret_func = log_regret, norm_data = True, num_runs = 1,step_size = 0.001,method = 'stein', k_stein = .1,  T =500, a = .1, R = 1, N= 10, q = 2, M = 1000, r_init = False, seeds = None, num_init = 10, true_min = 0, noise = False):
    
    print('Method: ', method, '\n')
#         kernel = Matern(length_scale=.1, nu = 2.5)
    if(noise):
        kernel = Matern(length_scale=.01, nu = 2.5) + WhiteKernel(noise_level=0.001, noise_level_bounds=(1e-8, 1))
    else:
        kernel = Matern(length_scale=.01, nu = 2.5)

    
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y = False,n_restarts_optimizer = 400)    
    
    num_experiments = len(seeds)
    regret = numpy.zeros((num_experiments, num_runs, iterations+1))
    dim = grid.n_dims
    obs_set = numpy.zeros((num_runs, num_init + iterations*N, grid.n_dims))
    obs_set_value = numpy.zeros((num_runs, num_init + iterations*N))
    
    if(method == 'stein'):
        return_gradient_method = return_grad_stein_combinations
    elif(method == 'wasserstein'):
        return_gradient_method = return_grad_wasserstein_combinations
    
    for index_seed, seed in enumerate(seeds):
        print('Seed = ', seed)
        
        
            
        it = 0
        if (r_init == True):
            numpy.random.seed(seed)
            lhs = Lhs(criterion="maximin", iterations=10000)
            obs_init = np.array(lhs.generate(grid.dimensions, num_init))
            value_init = func(obs_init)
            np.save('test.npy', obs_init)

        for run in range(num_runs):
            if(r_init == True):  

                obs_set[run, 0:num_init] =obs_init
                obs_set_value[run, 0:num_init] = value_init

            else:
                obs_set[run, 0:num_init] = set_init
                obs_set_value[run, 0:num_init] = set_init_val


            regret[index_seed, run,0] = (regret_func(obs_set[run, 0:num_init], obs_set_value[run, 0:num_init]))
        print('Regret start = ', regret_func(obs_set[run, 0:num_init], obs_set_value[run, 0:num_init]))
        while(it  < iterations):
            print('BO iteration = ', it + 1)
            kernel_params_list = []
            gpr_list = []
            data = []
            starting_points = []
            
            for run in range(num_runs):
                
                #normalise data
                if(norm_data):
                    norm_y = (obs_set_value[run, 0:num_init+(it*N)]- np.min(obs_set_value[run, 0:num_init+(it*N)]))/(np.max(obs_set_value[run, 0:num_init+(it*N)])-np.min(obs_set_value[run, 0:num_init+(it*N)]))
#                     norm_y = (obs_set_value[run, 0:num_init+(it*N)]- np.mean(obs_set_value[run, 0:num_init+(it*N)]))/(np.std(obs_set_value[run, 0:num_init+(it*N)]))

                        # Fit to data to get best kernel parameters 
                    gpr.fit(obs_set[run, 0:num_init+(it*N)], norm_y)
                    data.append(norm_y)
                else:
                    gpr.fit(obs_set[run, 0:num_init+(it*N)], obs_set_value[run, 0:num_init+(it*N)])
                    data.append(obs_set_value[run])

                #print(gpr.kernel_.hyperparameters)
                
#                 kernel_params = gpr.kernel_.length_scale
#                 kernel_params = gpr.kernel_.length_scale
                if(noise):
                    kernel_params = [gpr.kernel_.k1.length_scale, gpr.kernel_.k2.noise_level]
                else:
                    kernel_params = [gpr.kernel_.length_scale ,0]
                
                kernel_params_list.append(kernel_params)
                gpr_list.append(gpr)
                
                current_point = []
    
                numpy.random.seed(run)
                starting_point = np.array([generate_new_sample(grid, N)])

                starting_points.append(starting_point)
                
            data = np.array(data)
            starting_points = np.array(starting_points)
            kernel_params_list = np.array(kernel_params_list)
            
            points, final = next_points_qEI_opt(obs_set, data, kernel_GP, kernel_params_list, starting_points, T, a , R , q ,  M,  k_stein,step_size ,num_init, it, return_gradient_method) #version with R>1

            numpy.save('Optimisation_routines/'+ str(func.__name__) +'/gradient_control/gradient_norm_'+method+'_iteration' + str(it) +'_N_' + str(N) +'_q_'+ str(q)+ '_seed_'+str(seed)+'.npy', final[1])
            numpy.save('Optimisation_routines/'+ str(func.__name__) +'/trace_control/trace_'+method+'_iteration' + str(it) +'_N_' + str(N) +'_q_'+ str(q)+ '_seed_'+str(seed)+'.npy', final[0])


            
            for run in range(num_runs):
                    next_obs_val = func(points[run])


                    obs_set[run, num_init+(it*N): num_init+(it+1)*N] = points[run]
                    obs_set_value[run, num_init+(it*N): num_init+(it+1)*N] = next_obs_val


                    print('Regret run ', run, ' = ', regret_func(obs_set[run, 0:num_init+(it+1)*N], obs_set_value[run, 0:num_init+(it+1)*N]))
                    
                    regret[index_seed, run, it+1] = (regret_func(obs_set[run, 0:num_init+(it+1)*N], obs_set_value[run, 0:num_init+(it+1)*N]))
            it +=1
            
            np.save('Optimisation_routines/' + str(func.__name__) + '/results/regret_'+method+'_' + str(func.__name__) +'_N_' + str(N) +'_q_'+ str(q)+ '_seed_'+str(seed)+'.npy', regret)
            np.save('Optimisation_routines/' + str(func.__name__) + '/results/obs_'+method+'_' + str(func.__name__) +'_N_' + str(N) +'_q_'+ str(q)+ '_seed_'+str(seed)+'.npy', obs_set)



    return(regrets, obs_set, obs_set_value)

