import os
import json
import cv2
import pickle
import copy
from tqdm import tqdm
import pdb
import torch
from ..data.preprocess import bdd100k_get_vidnm2vidid
from ..data.normalize import img_mean_std, norm_imgs
from ..data.transforms import RandomHorizontalFlip, RandomResizedCrop, crop_local_ctx
from torchvision.transforms import functional as TVF


class BDD100kDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root='/home/y_feng/workspace6/datasets/BDD100k/bdd100k',
                 subsets='train_val',
                 ):
        super().__init__()
        self.data_root = data_root
        self.subsets = subsets
        self._subsets = self.subsets.split('_')
        self.extra_root = os.path.join(data_root, 
                                       'extra')
        vid_nm2id_path = os.path.join(self.extra_root, 
                                      'vid_nm2id.pkl')
        vid_id2nm_path = os.path.join(self.extra_root, 
                                      'vid_id2nm.pkl')
        if os.path.exists(vid_nm2id_path) and \
            os.path.exists(vid_id2nm_path):
            with open(vid_id2nm_path, 'rb') as f:
                self.vid_id2nm = pickle.load(f)
            with open(vid_nm2id_path, 'rb') as f:
                self.vid_nm2id = pickle.load(f)
        else:
            self.vid_id2nm, self.vid_nm2id = \
                bdd100k_get_vidnm2vidid(data_root=self.data_root,
                                        sub_set=self.subsets)
        
            


if __name__ == '__main__':
    pass
    img_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/'
    label_train_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20/train'
    label_val_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20/val'
    label_train_vid = os.listdir(label_train_root)
    label_val_vid = os.listdir(label_val_root)
    img_vid = os.listdir(img_root)
    print(len(label_train_vid), len(label_val_vid), len(img_vid))