import numpy as np
from numpy import sin, cos, tan

phi1s = np.arange(0.01, 2*np.pi, 0.03)
find_sin = lambda theta1, phi1, theta2, mu: ((mu/2) * (cos(2*theta1)*cos(2*theta2) -1) - (sin(2*theta2)*sin(phi1)*cos(theta1)**3)) / (sin(2*theta1)*cos(theta2)**3)
params_string = ""
ind = 1
theta1 = 0.9
theta2 = 5.41

for phi1 in phi1s:
    ID = str(ind)
    sinphi2 = find_sin(theta1, phi1, theta2, 0.325)
    if (sinphi2 > 1) | (sinphi2 < -1): continue; 
    params_string += ID
    params_string += f" {phi1} {np.arcsin(sinphi2)} {theta2}\n"
    ind += 1
        
print(params_string)
    

