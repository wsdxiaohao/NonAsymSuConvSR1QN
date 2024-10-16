#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:26:42 2024

@author: shidawang
"""
from Mathtools import *

# Cubic SR1 quasi Newton
def Cubic_SR1(Model, options, tol=0, maxiter=1000, check=10):
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
    #intial
    rkm = 0
    Skm = 0 #safeguard
    Gk  = L*np.identity(N)
    Gkp = Gk
    gradk = gradient(xk,A,b,mu)
    gradkp =gradk
    Hk = L*np.identity(N) #Hessian(xkp,A,b,mu);#hessian
    Hkp =Hk 
    #Hk = 1/L*np.identity(N)
    S_km = 0
    r_km = 0
    rk = 0
    S_k = 0
    lambdak = 0
    fk = (objective(xk, A, b,mu))[0,0]
    breakvalue =0 
    #time     
    time = 0
    seq_time =[0]
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
        Hk = Hkp
        Gk = Gkp
        gradk = gradkp
        S_km = S_k
        #compute lambdak
        normgradk = norm(gradk)
        
        #lk = (np.sqrt(H*normgradk)) #for test
        pk = np.linalg.solve((Gk), gradk)
        
        
        lambdak1 = 2*H*rkm #+ np.sqrt(2*H*normpk/mu)
        
       
        
        Gk_temp = Gk+lambdak1*np.identity(N)
        
        
       
        #Newton step
        g = gradk.reshape(len(gradk))
        x= xk.reshape(len(xk))
        xkp,_ = ls_cubic_solver(x, g, Gk_temp, H)
        xkp=xkp.reshape((len(xkp),1))
        rk = norm(xkp-xk)
        lambdak2 = np.sqrt(2*H*rk)
        
        #xkp = xk- np.linalg.solve((Hk+lambdak), gradk)
        #compute rk
        gradkp = gradient(xkp,A,b,mu)
        
        
        tGk = Gk_temp+ lambdak2*np.identity(N)
        Gkp =  SR1metric(xkp,xk,gradkp,gradk,tGk)#SR1
        
        
        
        #compute new value of smooth part of objective
        fkp = objective(xkp, A, b,mu);
        
        #time
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
    output={
            'sol': xkp,
            'seq_obj': seq_obj,
            'seq_time':seq_time,

            'breakvalue':breakvalue
            }
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
    if options['storeObjective'] == True:
        output['seq_fun'] = seq_obj;
    
    return output