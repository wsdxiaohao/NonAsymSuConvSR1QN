#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:04:43 2024

@author: shidawang
"""
from Mathtools import *
#import numpy as np
"""
# SR1 regularized quasi Newton
def Grad_SR1(Model, options, tol=0, maxiter=1000, check=10):
    
    #load Model
    xk = Model['x']
    xkp = xk
    A = Model['A']
    b = Model['b']
    mu = Model['mu']
    L = Model['Lip']
    H = Model['H']
    M = len(b)
    N = len(A[0,:])
    ##############################
    #bar kappa
    kappa = N*L
    ##############################
    
    #intial
    rkm = 0
    Skm = 0 #safeguard
    Gk  = L*np.identity(N)
    Gkp = Gk
    hatGkp = Gkp
    gradk = gradient(xk,A,b,mu)
    gradkp =gradk
    #Hk = 1/L*np.identity(N)
    r_km = 0
    rk = 0
    S= 0
    lambdak = 0
    fk = (objective(xk, A, b,mu))[0,0]
    breakvalue =0 
    num_restart = 0 
    # taping
    if options['storeResidual'] == True:
        #opt = options['optimal value']
        seq_res = [norm(gradk)]
    if options['storeObjective'] == True:
        seq_obj = np.zeros(maxiter+1);        
        seq_obj[0] = fk;
#     if options['storeTime'] == True:
#         seq_time = zeros(maxiter+1);
#         seq_time[0] = 0;
    time = 0
    seq_time=[0]
    for iter in range(1,maxiter+1):
        stime = clock.time();

        rkm = rk
        xk = xkp
        Gk = Gkp
        gradk = gradkp
        
        #Newton step
        #Newton step
        hatGk = hatGkp
        S = np.trace(hatGk)
        if S<kappa:
            tGk = hatGk
            xkp = xk- np.linalg.solve(hatGk,gradk)
        else:
            num_restart = num_restart + 1
            tGk   = L*np.identity(N)
            xkp = xk- gradk/L
            print('*****')
        #xkp = xk- np.linalg.solve((Gk+lambdak), gradk)
        #compute rk
        rk = norm(xkp-xk)
        gradkp = gradient(xkp,A,b,mu)
        normgradkp = norm(gradkp)

        
        Gkp = SR1metric(xkp,xk,gradkp,gradk,tGk)
        lambdakp = 2*H*rk/mu + np.sqrt(2*H*normgradkp/mu)

        hatGkp = (1+lambdak)*Gkp
        #compute new value of smooth part of objective
        fkp = objective(xkp, A, b,mu);
        
        time = time + (clock.time() - stime);
        #if (iter % check == 0):
        seq_time.append(time)
        #check breaking condition
        if options['storeResidual'] == True:
            res = norm(gradk)#fkp[0,0]-opt
        else:
            res = norm(gradk)#fkp[0,0]
        if res < tol:
            breakvalue = 2;
            
        
        #print info
        if (iter%check ==0):
            
            print('iter:%d, funValue:%f,fun:%f'%(iter,fkp[0],fkp[0]))
        #handle breaking condition
       
        # tape residual
        
        if options['storeResidual'] == True:
            seq_res.append(res);
        
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;
        if options['storeObjective'] == True:
            seq_obj[iter] = fkp[0,0];
        #if options['storeBeta'] == True:
        #    seq_beta[iter-1] = beta;
        if breakvalue == 2:
            print('Tolerence value reached');
            break
#return results
    print('num_restart=',num_restart)
    output={
            'sol': xkp,
            'seq_obj': seq_obj,
            'breakvalue':breakvalue,
            'seq_time':seq_time
            }
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
    if options['storeObjective'] == True:
        output['seq_fun'] = seq_obj;
    
    return output
"""


# SR1 regularized quasi Newton
def Grad_SR1(Model, options, tol, maxiter=1000, check=10):
    
    #load Model
    xk = Model['x']
    xkp = xk
    A = Model['A']
    b = Model['b']
    mu = Model['mu']
    L = Model['Lip']
    H = Model['H']
    M = len(b)
    N = len(A[0,:])
    ##############################
    #bar kappa
    kappa = N*L
    ##############################
    
    #intial
    rkm = 0
    Skm = 0 #safeguard
    Gk  = L*np.identity(N)
    invGk = 1/L*np.identity(N)
    invhatGkp = invGk
    Gkp = Gk
    invGkp = invGk
    hatGkp = Gkp
    gradk = gradient(xk,A,b,mu)
    gradkp =gradk
    #Hk = 1/L*np.identity(N)
    r_km = 0
    rk = 0
    S= 0
    lambdak = 0
    fk = (objective(xk, A, b,mu))[0,0]
    breakvalue =0 
    num_restart = 0
    
    ####################
    time = 0
    seq_time =[0]
    ####################
    # taping
    if options['storeResidual'] == True:
        #opt = options['optimal value']
        seq_res = [norm(gradk)]#np.zeros(maxiter+1);
        #seq_res[0] = fk - opt;
    if options['storeObjective'] == True:
        seq_obj = np.zeros(maxiter+1);        
        seq_obj[0] = fk;
    for iter in range(1,maxiter+1):
        #time
        stime = clock.time()
        
        
        rkm = rk
        xk = xkp
        #Gk = Gkp
        invhatGk = invhatGkp
        gradk = gradkp
        
        #Newton step
        #Newton step
        #hatGk = hatGkp
        DinvhatGk = np.diag(invhatGk)
        DhatGk = np.diag(1/DinvhatGk)
        S = np.trace(DhatGk)
        if S<kappa:
            #xkp = xk- np.linalg.solve(hatGk,gradk)   #Using sherman morrison woodburry formula
            xkp = xk - invhatGk@gradk
        else:
            hatGk   = L*np.identity(N)
            xkp = xk- gradk/L
            print('*****number of iterations***')
            num_restart = num_restart+1
            print(iter)
            
        #xkp = xk- np.linalg.solve((Gk+lambdak), gradk)
        #compute rk
        rk = norm(xkp-xk)
        gradkp = gradient(xkp,A,b,mu)
        normgradkp = norm(gradkp)

        
        invGkp = invSR1metric(xkp,xk,gradkp,gradk,invhatGk)
        
        lambdakp = 2*H*rk/mu + np.sqrt(2*H*normgradkp/mu)

        #hatGkp = (1+lambdak)*Gkp
        invhatGkp = invGkp/(1+lambdak)
        
        #compute new value of smooth part of objective
        fkp = objective(xkp, A, b,mu);
        #compute time 
        time = time + (clock.time() - stime);

        seq_time.append(time)
        #check breaking condition
        if options['storeResidual'] == True:
            res = norm(gradk)#fkp[0,0]-opt
        else:
            res = norm(gradk)#fkp[0,0]
        if res < tol:
            breakvalue = 2;
            
        
        #print info
        if (iter%check ==0):
            
            print('iter:%d, funValue:%f,fun:%f'%(iter,fkp[0],fkp[0]))
        #handle breaking condition
        if breakvalue == 2:
            print('Tolerence value reached');
        # tape residual
        
        if options['storeResidual'] == True:
            seq_res.append(res);
        
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;
        if options['storeObjective'] == True:
            seq_obj[iter] = fkp[0,0];
        #if options['storeBeta'] == True:
        #    seq_beta[iter-1] = beta;
        if breakvalue == 2:
            break
        
#return results
    print('num_restart=',num_restart)
    output={
            'sol': xkp,
            'seq_obj': seq_obj,
            'seq_time': seq_time,
            'breakvalue':breakvalue
            }
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
    if options['storeObjective'] == True:
        output['seq_fun'] = seq_obj;
    
    return output




































