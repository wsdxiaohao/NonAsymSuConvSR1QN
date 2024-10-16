#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:06:51 2024

@author: shidawang
"""




from Mathtools import *
from numpy import zeros, sqrt

def HBF(Model, options, tol, maxiter, check):
    """
    
    FISTA algorithm for solving:
        min_{x} f(x);  f(x)
    Update step:
        
        
        
    Properties:
    -----------
    f            convex, continuously differentiable with L-Lipschitz
    -----------
    mode        model data of the optimization problem
    
    
    options (required):
    -'stepsize' stepsize alpha = 1/L (backtracking can be used)
    -'init'     initialization
    
    options (optional):
    
    tol         tolerance threshold for the residual
    maxiter     maximal number of iterations
    check       provide information after 'check' iterations
    
    Return:
    -------
    output:
    -'sol'      solution of the problems
    -'seq_res'  sequence of residual values (if activiated)
    -'seq_time' sequence of time points (if activiated)
    -'seq_x'    sequence of iterates (if activiated)
    -'seq_obj'  sequence of objective values (if activiated)
    -'seq_beta' sequence of beta values ( )
    -'breakvalues' code for the type of breaking condition
                   1: maximal number of iterations exceeded
                   2: breaking condition reached (residual below tol)
                   3: not enough backtracking iteration
    """
    
    #store options
    #backtracking options
    
    #load oracle
    
    
    #load parameter
    tau   = options['stepsize'];
    
    mu    = Model['mu']
    Lip   = Model['Lip']
    A     = Model['A']
    b     = Model['b']
    # initalization
    x_kp1 = Model['x'];
    x_k   = x_kp1.copy()
    x_km  = x_k.copy()
    y_k   = x_kp1.copy()
    t_k   = 1;
    f_kp1 = objective(x_kp1, A, b,mu)
    gradk = gradient(x_k, A, b,mu)
    #res0  = residual(x_kp1, 1.0, model,options);
    
    
    sqrtL = np.sqrt(Lip)
    sqrtmu = np.sqrt(mu)
    
    # taping
    # taping
    if options['storeResidual'] == True:
        #seq_res = np.zeros(maxiter+1);
        #opt = options['optimal value']
        seq_res = [norm(gradk)]#f_kp1[0,0]-opt;
    if options['storeTime'] == True:
        #seq_time = zeros(maxiter+1);
        seq_time=[0];
    if options['storePoints'] == True:
        seq_x = np.zeros((model['N'],maxiter+1));        
        seq_x[:,0] = x_kp1;
    if options['storeObjective'] == True:
        seq_obj = np.zeros(maxiter+1);        
        seq_obj[0] = f_kp1[0];
#     if options['storeBeta'] == True:
#         seq_beta = zeros(maxiter);        
    time = 0;
    # solve
    breakvalue = 1;
    
    n=len(x_k) -1
    for iter in range(1,maxiter+1):
        stime = clock.time();
        
        # update variables
        t_kp1 = 0.5*(1.0+ sqrt(1.0 + 4*t_k**2));#(iter +2.0)/3
        beta  = float((sqrtL-sqrtmu)/(sqrtL+sqrtmu));
        tau   = 4/((sqrtL+sqrtmu)**2)
        
        x_km = x_k
        
        x_k = x_kp1.copy();
        f_k = f_kp1.copy();
        
        # compute gradient
        grad_k = gradk#gradient(y_k, A, b,mu)
        #forward step
        x_kp1 = x_k -tau * grad_k + beta*(x_k-x_km)
        #backward step
        
        #x_kp1[:n] = prox_g(x_kp1[:n],tau,model,options);
        #compute new value of smooth part of objective
        f_kp1 = objective(x_kp1, A, b,mu)
        #compute new objective value
        
        
        
        time = time + (clock.time() - stime);

        #check breaking condition
        gradk = gradient(x_kp1, A, b,mu)

        
        if options['storeResidual'] == True:
            res = norm(grad_k)#f_kp1[0,0]-opt
        else:
            res = norm(grad_k)#f_kp1[0,0]
        
        if res < tol:
            breakvalue = 2;
            
        #print info
        if (iter%check ==0):
            
            print('iter:%d, time:%5f, tau:%f,fun:%f'%(iter,stime,tau,f_kp1[0]))
        #handle breaking condition
        
        # tape residual
        if options['storeResidual'] == True:
            seq_res.append(res);
        if options['storeTime'] == True:
            seq_time.append(time);
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;
        if options['storeObjective'] == True:
            seq_obj[iter] = f_kp1[0,0];
        if options['storeBeta'] == True:
            seq_beta[iter-1] = beta;
        if breakvalue == 2:
            print('Tolerence value reached');
            break
        
#return results
    output={
            'sol': x_kp1,
            'seq_obj': seq_obj,
            'seq_time':seq_time,
            'breakvalue':breakvalue
            }
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
    return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    