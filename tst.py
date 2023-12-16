import pickle
import os
import pdb
import numpy as np

path = '/home/y_feng/workspace6/datasets/PIE_dataset/seg/set01/video_0001/01013.pkl'
with open(path, 'rb') as f:
    a = pickle.load(f)
print(a.shape)
print(np.max(a), np.min(a))

def main():
    b = 'img'
    print(globals())
    print(locals())
    print(globals()['b'])


if __name__ == '__main__':
    main()