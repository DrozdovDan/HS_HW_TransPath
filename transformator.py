import numpy as np
import os

data = np.load(f'./output_tensor.npy', mmap_mode='c')

maps = np.expand_dims(data[..., 0] == 0, 1)
starts = np.expand_dims(data[..., 1] == 2, 1)
goals = np.expand_dims(data[..., 1] == 8, 1)
focal = np.zeros_like(maps).astype(bool)
cf = np.zeros_like(maps).astype(bool)
abs = np.zeros_like(maps).astype(bool)

os.makedirs(f'./eval_data/eval', exist_ok=True)
np.save(f'./eval_data/eval/maps.npy', maps)
np.save(f'./eval_data/eval/starts.npy', starts)
np.save(f'./eval_data/eval/goals.npy', goals)
np.save(f'./eval_data/eval/focal.npy', focal)
np.save(f'./eval_data/eval/cf.npy', cf)
np.save(f'./eval_data/eval/abs.npy', abs)
