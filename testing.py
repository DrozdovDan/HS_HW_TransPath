import numpy as np
import matplotlib.pyplot as plt

maps1 = np.load('./TransPath_data/train/maps.npy', mmap_mode='c')
maps2 = np.load('./TransPath_data/val/maps.npy', mmap_mode='c')
maps3 = np.load('./TransPath_data/test/maps.npy', mmap_mode='c')

print(maps1.shape)
print(maps2.shape)
print(maps3.shape)