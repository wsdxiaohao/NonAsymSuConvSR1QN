#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:00:03 2024

@author: shidawang
"""
import numpy as np
import matplotlib.pyplot as plt
import time as clock
from libsvm.svmutil import *

# To compute the value of the objective, we implement the following python function.
def objective(x, A, b,mu):
    # 
    M=len(A)
    err = (-1)*b*(A @ x)
    obj = np.sum(np.log(1+np.exp(err ) ))/M + 0.5*mu*x.T@x
    return obj

# To compute the gradient of the objective, 
def gradient(x, A, b,mu):
    M = len(A)
    r=(-1)*b*(A@x)
    ex = np.exp(r)
    #grad = A.T@( (-1)*b*(ex/(1+ex)))/M
    grad = A.T@( (-1)*b*(1/(1+(1/ex))))/M
    #grad = np.reshape(grad, (len(grad),1)) 
    grad = grad + mu*x
    return grad

# To compute the Hessian of the objective
def Hessian(x,A,b,mu):#wrong
    M=len(A)
    r = (-1)*b*(A@x)
    p = 1/(1+np.exp(r))
    Hessian = A.T@((p*(1-p))*A)/M+mu*np.identity(len(A[0,:]))
    
    return Hessian

# To compute the SR1 update
def SR1metric(xkp,xk,gradkp,gradk,tGk,option=1):
    #input diagnal matrix lambdak*identity
    sk = xkp-xk
    yk = gradkp-gradk
    row = yk-tGk@sk
    den = (row.T@sk)[0,0]
    Gkp  = tGk +row@row.T/den
    test = np.abs(den/((sk.T@sk)[0,0]))
    if test<1e-8:
        return tGk
    else:
        return Gkp
    
# To update the inverse of SR1 metric
def invSR1metric(xkp,xk,gradkp,gradk,invtGk):
    #input diagnal matrix lambdak*identity
    sk = xkp-xk
    yk = gradkp-gradk
    row = invtGk@yk-sk
    den = (-row.T@yk)[0,0]
    invGkp  = invtGk + row@row.T/den
    test = np.abs(den/((sk.T@sk)[0,0]))
    if test<1e-8:
        return invtGk
    else:
        return invGkp
    
#
def norm(v):
    norm = np.sqrt(v.T@v)[0]
    return norm


def lanorm(v):
    norm = np.sqrt(np.sum(v**2))
    return norm












"From Nicolas Mishenko https://github.com/konstmish/opt_methods/blob/master/optmethods/second_order/cubic.py"

from Mathtools import *


def ls_cubic_solver(x, g, H, M, it_max=100, epsilon=1e-8, loss=None):
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3
    
    For explanation of Cauchy point, see "Gradient Descent 
        Efficiently Finds the Cubic-Regularized Non-Convex Newton Step"
        https://arxiv.org/pdf/1612.00547.pdf
    Other potential implementations can be found in paper
        "Adaptive cubic regularisation methods"
        https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf
    """
    solver_it = 1
    newton_step = -np.linalg.solve(H, g)
    if M == 0:
        return x + newton_step, solver_it
    def cauchy_point(g, H, M):
        if lanorm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / lanorm(g)
        H_g_g = H @ g_dir @ g_dir
        R = -H_g_g / (2*M) + np.sqrt((H_g_g/M)**2/4 + lanorm(g)/M)
        return -R * g_dir
    
    def conv_criterion(s, r):
        """
        The convergence criterion is an increasing and concave function in r
        and it is equal to 0 only if r is the solution to the cubic problem
        """
        s_norm = lanorm(s)
        return 1/s_norm - 1/r
    
    # Solution s satisfies ||s|| >= Cauchy_radius
    r_min = lanorm(cauchy_point(g, H, M))
    
    if loss is not None:
        x_new = x + newton_step
        if loss.value(x) > loss.value(x_new):
            return x_new, solver_it
        
    r_max = lanorm(newton_step)
    if r_max - r_min < epsilon:
        return x + newton_step, solver_it
    id_matrix = np.eye(len(g))
    for _ in range(it_max):
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if np.abs(crit) < epsilon:
            return x + s_lam, solver_it
        if crit < 0:
            r_min = r_try
        else:
            r_max = r_try
        if r_max - r_min < epsilon:
            break
    return x + s_lam, solver_it

























