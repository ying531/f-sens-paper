#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:13:42 2022

Code for experiments in Section 5.2
Bounds on ATC for varying rho with a fixed value of delta=0.5

@author: ying
"""

import numpy as np
import pandas as pd 
import random
from patsy import dmatrix 
from sklearn.ensemble import RandomForestRegressor 
from scipy.optimize import Bounds, minimize
from utils import gen_data, fc, fcp, fcpp, floss, grad, hessian, loss, turn_spline, opt_KL, ifold
import os
import sys

seed = int(sys.argv[1])
rho_id = int(sys.argv[2]) 

# import mkl
# mkl.set_num_threads(1)
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

save_dir = './fix_results/'
os.makedirs(save_dir, exist_ok=True)


n = 15000
p = 4
beta1 = np.array((0.531, 1.126, -0.312, 0.671))
beta0 = np.array((-0.531, -.126, -0.312, 0.671))
gamma = np.array((-0.531, 0.126, -0.312, 0.018))

delta = 0.5
rhos = np.linspace(0.005, 0.25, num = 50)

 

rho = rhos[rho_id]

# =============================================================================
# # generate and split data
# =============================================================================

random.seed(seed)
data = gen_data(n, p, delta, beta1, beta0, gamma)

t_idx = np.array(range(n))[data["T"]==1]
c_idx = np.array(range(n))[data["T"]==0]
nt = len(t_idx)
nc = len(c_idx)

# split data, 3 folds
tidx = random.sample(range(nt), nt)
tidx_list = [t_idx[tidx[0:int(np.floor(nt/3))]], 
             t_idx[tidx[int(np.floor(nt/3)):int(np.floor(2*nt/3))]],
             t_idx[tidx[int(np.floor(2*nt/3)):nt]]]

cidx = random.sample(range(nc), nc)
cidx_list = [c_idx[cidx[0:int(np.floor(nc/3) )]],
             c_idx[cidx[int(np.floor(nc/3)):int(np.floor(2*nc/3))]],
             c_idx[cidx[int(np.floor(2*nc/3)):nc]]]


knots = (0.25, 0.5, 0.75)
pp = np.mean(data["T"])
 

all_mu_upp = np.zeros(3)
all_mu_low = np.zeros(3)
# influence function
inf_upp = np.zeros(n)
inf_low = np.zeros(n)

# =============================================================================
# # lower bound
# =============================================================================
 
for j in range(3):
    ### train on fold j
    # train e(x)
    aidx1 = np.concatenate((tidx_list[ifold(j,3)], cidx_list[ifold(j,3)]))
    ex_rf = RandomForestRegressor().fit(data["X"][aidx1,], data["T"][aidx1])
    
    # optimize hat alpha, hat eta 
    eps = 0.01 
    # randomly initialize for 3 times
    all_probs = []
    all_losses = []
    for rrr in range(3):
        this_run = opt_KL(data["X"][tidx_list[ifold(j,3)],], 
                      data["Y1"][tidx_list[ifold(j,3)]], rho, eps, knots)
        all_probs.append(this_run)
        all_losses.append(this_run["val"])
        # print(all_losses)
    opt_prob = all_probs[np.argmin(all_losses)]

    ### regress on fold j+1
    new_Xdat = turn_spline(data["X"][tidx_list[ifold(j+1,3)],], knots)
    new_ax = np.maximum(eps, new_Xdat @ opt_prob["alpha"])
    new_eta = new_Xdat @ opt_prob["eta"]
    Hval = new_ax * fc(new_ax, new_eta, data["Y1"][tidx_list[ifold(j+1,3)]], rho) + new_eta + new_ax * rho
    reg_rf = RandomForestRegressor().fit(data["X"][tidx_list[ifold(j+1,3)],], Hval)
    
    ### adjust on fold j+2
    adj_Xdat = turn_spline(data["X"][tidx_list[ifold(j+2,3)],], knots)
    adj_ax = np.maximum(eps, adj_Xdat @ opt_prob["alpha"])
    adj_eta = adj_Xdat @ opt_prob["eta"]
    adj_Hval = adj_ax * fc(adj_ax, adj_eta, data["Y1"][tidx_list[ifold(j+2,3)]], rho) + adj_eta + adj_ax * rho
    adj_h1 = reg_rf.predict(data["X"][tidx_list[ifold(j+2,3)],])
    adj_h0 = reg_rf.predict(data["X"][cidx_list[ifold(j+2,3)],])
    adj_ex = ex_rf.predict(data["X"][tidx_list[ifold(j+2,3)],])
    adj_rat = (1-adj_ex) * pp / (adj_ex * (1-pp))
    
    inf_low[tidx_list[ifold(j+2,3)]] = adj_rat * (adj_Hval - adj_h1) / pp
    inf_low[cidx_list[ifold(j+2,3)]] = adj_h0 / (1-pp) + data["Y0"][cidx_list[ifold(j+2,3)]]
    
    mu_j = np.mean(adj_rat * (adj_Hval - adj_h1)) + np.mean(adj_h0)
    all_mu_low[j] = mu_j

hat_mu_low = - np.mean(all_mu_low)


# =============================================================================
# # upper bound
# =============================================================================

for j in range(3):
    ### train on fold j
    # train e(x)
    aidx1 = np.concatenate((tidx_list[ifold(j,3)], cidx_list[ifold(j,3)]))
    ex_rf = RandomForestRegressor().fit(data["X"][aidx1,], data["T"][aidx1])
    
    # optimize hat alpha, hat eta 
    eps = 0.01 
    # randomly initialize for 3 times
    all_probs = []
    all_losses = []
    for rrr in range(3):
        this_run = opt_KL(data["X"][tidx_list[ifold(j,3)],], 
                      -data["Y1"][tidx_list[ifold(j,3)]], rho, eps, knots)
        all_probs.append(this_run)
        all_losses.append(this_run["val"])
    opt_prob = all_probs[np.argmin(all_losses)]

    ### regress on fold j+1
    new_Xdat = turn_spline(data["X"][tidx_list[ifold(j+1,3)],], knots)
    new_ax = np.maximum(eps, new_Xdat @ opt_prob["alpha"])
    new_eta = new_Xdat @ opt_prob["eta"]
    Hval = new_ax * fc(new_ax, new_eta, -data["Y1"][tidx_list[ifold(j+1,3)]], rho) + new_eta + new_ax * rho
    reg_rf = RandomForestRegressor().fit(data["X"][tidx_list[ifold(j+1,3)],], Hval)
    
    ### adjust on fold j+2
    adj_Xdat = turn_spline(data["X"][tidx_list[ifold(j+2,3)],], knots)
    adj_ax = np.maximum(eps, adj_Xdat @ opt_prob["alpha"])
    adj_eta = adj_Xdat @ opt_prob["eta"]
    adj_Hval = adj_ax * fc(adj_ax, adj_eta, -data["Y1"][tidx_list[ifold(j+2,3)]], rho) + adj_eta + adj_ax * rho
    adj_h1 = reg_rf.predict(data["X"][tidx_list[ifold(j+2,3)],])
    adj_h0 = reg_rf.predict(data["X"][cidx_list[ifold(j+2,3)],])
    adj_ex = ex_rf.predict(data["X"][tidx_list[ifold(j+2,3)],])
    adj_rat = (1-adj_ex) * pp / (adj_ex * (1-pp))
    
    inf_upp[tidx_list[ifold(j+2,3)]] = adj_rat * (adj_Hval - adj_h1) / pp
    inf_upp[cidx_list[ifold(j+2,3)]] = adj_h0 / (1-pp) - data["Y0"][cidx_list[ifold(j+2,3)]]
    
    mu_j = np.mean(adj_rat * (adj_Hval - adj_h1)) + np.mean(adj_h0)
    all_mu_upp[j] = mu_j

hat_mu_upp = np.mean(all_mu_upp)

true_cmean = np.mean(data["Y1"][data["T"]==0])
sd_low = np.std(inf_low)
sd_upp = np.std(inf_upp) 
summary = pd.DataFrame({"hat_low": [hat_mu_low], "hat_upp": [hat_mu_upp],
                        "sd_low": [sd_low], "sd_upp": [sd_upp],
                        "nt": [nt], "nc": [nc], "true_cmean": true_cmean,
                        "seed": [seed], "rho": [rho]})
    
   
     
summary.to_csv(os.path.join(save_dir, "sum_"+str(seed)+"_rho_"+str(rho_id)+".csv"))





