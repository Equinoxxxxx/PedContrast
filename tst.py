import pickle
import os
import pdb
import cv2
import torch
import numpy as np
import json

from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.nuscenes import NuscDataset

# titan = TITAN_dataset(sub_set='default_test', norm_traj=1,
#                                       obs_len=16, pred_len=1, overlap_ratio=1, 
#                                       required_labels=[
#                                                         'atomic_actions', 
#                                                         'simple_context', 
#                                                         'complex_context', 
#                                                         'communicative', 
#                                                         'transporting',
#                                                         'age'
#                                                         ], 
#                                       multi_label_cross=1,  
#                                     #   use_cross=1,
#                                       use_atomic=1, 
#                                       use_complex=1, 
#                                       use_communicative=1, 
#                                       use_transporting=1, 
#                                       use_age=1,
#                                       tte=None,
#                                       small_set=0,
#                                       modalities='img_sklt_ctx_traj_ego',
#                                       ctx_format='ori_local',
#                                       augment_mode='random_crop_hflip'
#                                       )

# pie = PIEDataset(dataset_name='PIE', seq_type='crossing',
#                     obs_len=15, pred_len=1, obs_interval=1,
#                     do_balance=False, subset='train', bbox_size=(224, 224), 
#                     img_norm_mode='torch', color_order='BGR',
#                     resize_mode='even_padded', 
#                     modalities='img_sklt_ctx_traj_ego',
#                     ctx_format='ori_local',
#                     augment_mode='random_crop_hflip',
#                     small_set=0,
#                     overlap_retio=0.6,
#                     tte=[0, 60],
#                     recog_act=0,
#                     normalize_pos=0,
#                     ego_accel=1,
#                     speed_unit='m/s')

# nusc = NuscDataset()

img_train_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/images/track/train'
label_train_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20/train'
label_val_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20/val'

allcls = set()
for lfnm in os.listdir(label_train_root):
    l_path = os.path.join(label_train_root, lfnm)
    with open(l_path) as f:
        vid_l = json.load(f)
    for img_l in vid_l:
        img_nm = img_l['name']
        img_id = img_nm.split('-')[-1].replace('.jpg', '')
        img_id_int = int(img_id)
        for l in img_l['labels']:
            cls = l['category']
            allcls.add(cls)

print(allcls)