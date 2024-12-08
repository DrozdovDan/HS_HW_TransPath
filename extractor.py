import numpy as np
import os

states = ['train', 'val', 'test']

for state in states:

    data_names = ['maps', 'starts', 'goals', 'focal', 'cf', 'abs']
    data = {}

    for name in data_names:
        data[name] = np.load(f'./TransPath_data/{state}/{name}.npy', mmap_mode='c')

    num_elements = 64000 if state == 'train' else 16000
    os.makedirs(f'./TransPath_data_mini/{state}', exist_ok=True)

    for name in data_names:
        np.save(f'./TransPath_data_mini/{state}/{name}.npy', data[name][:num_elements])
