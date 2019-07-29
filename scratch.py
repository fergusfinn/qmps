import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

gs = np.linspace(0, 2, 10)

ex = np.load('exact.npy')
mp = np.load('calc.npy')

f = interp1d(gs, mp, kind='linear', fill_value='extrapolate')

plt.plot(gs, ex)
plt.plot(gs, mp)

gs = np.linspace(0, 2, 100)
plt.plot(gs, f(0.8*gs))
plt.show()
