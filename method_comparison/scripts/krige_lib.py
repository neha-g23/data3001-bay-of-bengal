import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
def xy(lat,lon): return np.c_[(lon-88)*111.32*np.cos(np.radians(14)), (lat-14)*110.57]
def gp(nu): return GPR(C(0.05,(1e-4,1))*Matern(100,(10,2000),nu=nu)+WhiteKernel(0.02,(1e-4,1)),normalize_y=True,n_restarts_optimizer=1,random_state=0)
