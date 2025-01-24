import numpy as np
import matplotlib.pyplot as plt
import os

main_dir = 'AlekSet'
folders = ['train', 'test', 'val']
dataset_dir = f'TransPath_{main_dir}'

for folder in folders: 
    print(f'Start transforming {folder}...')

    data = np.load(f'./{main_dir}/{folder}/maps.npy', mmap_mode='c')
    maps = np.expand_dims(data[..., 0] == 0, 1)
    starts = np.expand_dims(data[..., 3] == 1, 1)
    goals = np.expand_dims(data[..., 1] == 1, 1)
    focal = np.zeros_like(maps).astype(bool)
    cf = np.load(f'./{main_dir}/{folder}/cf.npy', mmap_mode='c')
    cf = np.expand_dims(cf, 1)
    abs = np.expand_dims(data[..., 2], 1)

    print(f'Start saving {folder}...')
    os.makedirs(f'./{dataset_dir}/{folder}', exist_ok=True)

    np.save(f'./{dataset_dir}/{folder}/maps.npy', maps)
    np.save(f'./{dataset_dir}/{folder}/starts.npy', starts)
    np.save(f'./{dataset_dir}/{folder}/goals.npy', goals)
    np.save(f'./{dataset_dir}/{folder}/focal.npy', focal)
    np.save(f'./{dataset_dir}/{folder}/cf.npy', cf)
    np.save(f'./{dataset_dir}/{folder}/abs.npy', abs)

    print(f'Finish saving {folder}!')
