
import numpy as np
import pandas as pd 
import random
from patsy import dmatrix 
from sklearn.ensemble import RandomForestRegressor 
from scipy.optimize import Bounds, minimize


def gen_data(n, p, delta, beta1, beta0, gamma):
    # generate covariates
    X = np.array(np.random.uniform(size = n*p)).reshape((n,p)) 
    sig_x = np.sqrt( 1+(2.5*X[:,0])**2 *0.5 )
    # generate treatment
    e_x = np.exp(X@gamma) / (1+np.exp(X@gamma))
    # generate treated and control samples
    T = np.random.binomial(1, p=e_x)
    # generate U
    U1 = np.random.normal(size = n)
    U0 = np.random.normal(size = n)
    # generate Y(1) Y(0) for T=1
    Y11 = X@beta1 + U1 * sig_x
    Y01 = X@beta0 + U1 * sig_x
    # generate Y(1) Y(0) for T=0
    Y10 = X@beta1 - delta * sig_x + U0 * sig_x
    Y00 = X@beta0 - delta * sig_x + U0 * sig_x
    # assemble
    Y1 = Y11
    Y1[T==0] = Y10[T==0]
    Y0 = Y00
    Y0[T==1] = Y01[T==1]
    U = U1
    U[T==0] = U0[T==0]
    
    ORY = np.exp(-delta* (Y1 - X@beta1) / (2*sig_x) - delta**2 / (2*sig_x**2))
    exu = ORY * e_x
        
    data = {"Y1": Y1, "Y0": Y0, "X": X, "T":T, "U": U, "ex": e_x, "exu": exu}
    return data
    
    

# conjugate f*(y+eta/(-alpha)) 
def fc(a_x, eta_x, y, rho, div = 'KL'):
    if div == 'KL':
        conj = np.exp(- (y + eta_x)/a_x-1) 
    return(conj)

# f*'(y+eta/(-alpha))
def fcp(a_x, eta_x, y, rho, div = 'KL'):
    if div == 'KL':
        conj = np.exp(- (y + eta_x)/a_x-1)
        
    return(conj)

# f*'’(y+eta/(-alpha))
def fcpp(a_x, eta_x, y, rho, div = 'KL'):
    if div == 'KL':
        conj = np.exp(- (y + eta_x)/a_x-1)
        
    return(conj)
        

def floss(a_x, eta_x, y, rho, div = 'KL'):
    loss = np.mean(a_x * fc(a_x, eta_x, y, rho,div) + eta_x + a_x * rho) 
    return(loss)

def grad(a_x, eta_x, y, rho, div = 'KL'):
    gradd = np.zeros(2)
    gradd[0] = np.mean(fc(a_x, eta_x,y,rho,div)) + rho + np.mean((y + eta_x) * fcp(a_x, eta_x,y,rho,div) / a_x ) 
    gradd[1] = 1 - np.mean( fcp(a_x, eta_x,y,rho,div) )
    return(gradd)

def hessian(a_x, eta_x, y, rho, div = 'KL'):
    hess = np.zeros((2,2))
    hess[0,0] = np.mean( (y+eta_x)**2 * fcpp(a_x, eta_x,y,rho,div) / a_x**3) 
    hess[1,1] = np.mean( fcpp(a_x, eta_x,y,rho,div) /a_x)  
    hess[1,0] = - np.mean( (y+ eta_x) * fcpp(a_x, eta_x,y,rho,div) / a_x**2 ) 
    hess[0,1] = hess[1,0]
    return(hess)


def loss(theta, Xdat, Y, rho, eps):
    p = int(len(theta)/2)
    alpha = theta[0:p]
    eta = theta[p:2*p]
    a_x =  np.maximum(eps, Xdat @ alpha)
    eta_x = Xdat @ eta
    loss = np.mean(a_x * fc(a_x, eta_x, Y, rho, div="KL") + eta_x + a_x * rho)
    return(loss)
    

def turn_spline(X, knots):
    Xdat = pd.DataFrame()
    for jcol in range(X.shape[1]):
        if jcol == 0:
            Xdat = pd.concat( (Xdat, dmatrix("bs(train, knots=knts, degree=3, include_intercept=True)", {"train": X[:,jcol], "knts": knots}, return_type='dataframe')), axis=1)
        else:
            Xdat = pd.concat( (Xdat, dmatrix("bs(train, knots=knts, degree=3, include_intercept=False)", {"train": X[:,jcol], "knts": knots}, return_type='dataframe')), axis=1)
    return(Xdat)


def opt_KL(X, Y, rho, eps, knots = (0.25, 0.5, 0.75)):
    # turn x into spline 
    Xdat = turn_spline(X, knots)
    
    alpha = np.random.uniform(0.02, 0.1, size = Xdat.shape[1])
    eta = np.random.uniform(-2, 2, size = Xdat.shape[1])
    theta = np.concatenate((alpha,eta))
      
    opt_prob = minimize(loss, theta, method='Nelder-Mead', 
             args = (Xdat, Y, rho, eps), options={"xtol":1e-4, "maxfev":200000, "disp":False})
    
    opt_theta = opt_prob.x 
    opt_alpha = opt_theta[0:len(alpha)]
    opt_eta = opt_theta[len(alpha):2*len(alpha)]
    opt_a_x =  np.maximum(eps, Xdat @ opt_alpha)
    opt_eta_x = Xdat @ opt_eta 
    
    
    return {"val": - loss(opt_theta, Xdat, Y, rho, eps), "theta": opt_theta, 
            "Xdat": Xdat, 
            "alpha": opt_alpha, "eta": opt_eta,
            "grad": grad(opt_a_x, opt_eta_x, Y, rho)}


def ifold(j, Nfold):
    return np.mod(j, Nfold)
