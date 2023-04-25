# This is a trial for SINDy algorithm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso

from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
import pysindy as ps
# bad code but allows us to ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Train of the LORENTZ MODEL

dt = .002
t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(lorenz, t_train_span, x0_train,
                    t_eval=t_train, **integrator_keywords).y.T
# Generate measurement data
dt = .003
feat_name = ['x','y','z']
opt = ps.STLSQ(threshold= 0)#chooses an algorithm to solve the problem

# now we create the actual SINDy model
model = ps.SINDy(feature_names=feat_name, optimizer=opt)
model.fit(x_train,t=dt)
model.print()