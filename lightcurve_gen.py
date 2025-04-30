import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy.special import jn
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.special as sps

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#=== 5PN AAK imports ================

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import Pn5AAKWaveform
from few.utils.utility import get_fundamental_frequencies


sys.path.append(os.path.abspath('../KerrOrbitalAngleConversion/build/lib.macosx-11.0-arm64-cpython-313/'))
import kerrangleconversions as kac

use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 1,  # we don't want a sparsely sampled trajectory
    "max_init_len": int(1e6),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

# set initial parameters (default parameters in FEW AAK Documentation)
traj = EMRIInspiral(func='pn5')

# Make the gravitational wave strain, 3D trajectory, crossings and EM signal
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, m, a, p0, e0, Y0,
                               Phi_phi0 = Phi_phi0, Phi_theta0 = Phi_theta0, Phi_r0 = Phi_r0,
                               dt=dt, T=T,upsample=True,fix_t=True)

#AAK_out = wave_generator(M, m, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich,
#                         Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, 
#                         dt=dt, T=T)
trajectory = bl2kep2(t, a, p, e, x, Phi_r, Phi_theta, Phi_phi)

intersections = intersect_trajectory_with_rotating_disk(trajectory, disk_center0, disk_normal0, disk_radius, omega)

lightcurve = gaussian_peaks(trajectory[0], intersections.T[0],width=dt*2)

# Saving to file
np.savetxt("crossing_times.txt",intersections.T[0])
#np.savetxt("strainseries.txt",np.array([np.arange(len(AAK_out))*dt,AAK_out.real,AAK_out.imag]).T)
np.savetxt("lightcurve.txt",np.array([trajectory[0],lightcurve]).T)
