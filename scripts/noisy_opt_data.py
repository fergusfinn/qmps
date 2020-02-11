import numpy as np
from matplotlib import pyplot as plt
plt.style.use('pub_fast')

for i in range(1, 6):
    x = np.load(f'{i}.npy')
    plt.plot(x)
plt.show()
