import numpy as np
import matplotlib.pyplot as plt

data_names = ['maps', 'starts', 'goals', 'focal', 'cf', 'abs']
data = {}

for name in data_names:
    data[name] = np.load(f'./TransPath_data/test/{name}.npy', mmap_mode='c')
    plt.imshow(data[name][0][0])
    plt.show()