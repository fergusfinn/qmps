import numpy as np

theta1s = np.arange(0.01, 2*np.pi, 0.3)
x,y = np.meshgrid(theta1s, theta1s, indexing = 'ij')
params_string = ""
ind = 1

for i in range(len(x)):
    for j in range(len(y)):
        ID = str(ind)
        params_string += ID
        params_string += f" {x[i,j]} 0 {y[i,j]}\n"
        ind += 1
        

with open('/home/ucapjmd/Scratch/inputs/params_3_site.txt', 'w') as file:    
    file.write(params_string)
    

