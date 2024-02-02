import pickle
import os
import pdb
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import pytorch_warmup as warmup

from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.nuscenes_dataset import NuscDataset
from tools.datasets.bdd100k import BDD100kDataset
from tools.data.coord_transform import nusc_3dbbox_to_2dbbox
from tools.utils import seed_all
from config import dataset_root
from tools.data.crop_images import crop_ctx

from models.PCPA import PCPA
from models.ped_graph23 import PedGraph

seed_all(42)
small_set = 0
ctx_format = 'ped_graph'
overlap_ratio=0.6
obs_len = 4
pred_len = 1
obs_fps = 2
tte = [0, int((obs_len+pred_len)/obs_fps*30+2)]
# titan = TITAN_dataset(sub_set='default_test', norm_traj=1,
#                       obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio, 
#                       obs_fps=obs_fps,
#                       required_labels=[
#                                         'atomic_actions', 
#                                         'simple_context', 
#                                         'complex_context', 
#                                         'communicative', 
#                                         'transporting',
#                                         'age'
#                                         ], 
#                       multi_label_cross=0,  
#                       use_cross=1,
#                       use_atomic=1, 
#                       use_complex=0, 
#                       use_communicative=0, 
#                       use_transporting=0, 
#                       use_age=0,
#                       tte=None,
#                       modalities=['img','sklt','ctx','traj','ego'],
#                       sklt_format='coord',
#                       ctx_format=ctx_format,
#                       augment_mode='random_crop_hflip',
#                       small_set=small_set,
#                       )

# pie = PIEDataset(dataset_name='PIE', seq_type='crossing',
#                   obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
#                   obs_fps=obs_fps,
#                   do_balance=False, subset='train', bbox_size=(224, 224), 
#                   img_norm_mode='torch', color_order='BGR',
#                   resize_mode='even_padded', 
#                   modalities=['img','sklt','ctx','traj','ego'],
#                   sklt_format='coord',
#                   ctx_format=ctx_format,
#                   augment_mode='random_crop_hflip',
#                   tte=tte,
#                   recog_act=0,
#                   normalize_pos=0,
#                   speed_unit='m/s',
#                   small_set=small_set,
#                   )

# nusc = NuscDataset(sklt_format='coord',
#                    ctx_format=ctx_format,
#                    small_set=small_set,
#                    obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
#                    obs_fps=obs_fps,)
# bdd = BDD100kDataset(subsets='train',
#                      sklt_format='coord',
#                      ctx_format=ctx_format,
#                      small_set=small_set,
#                      obs_len=obs_len, pred_len=pred_len, overlap_ratio=overlap_ratio,
#                      obs_fps=obs_fps,)
# cat_set = torch.utils.data.ConcatDataset([
#     nusc, 
#     bdd, 
#     pie, 
#     titan
#     ])
# loader = torch.utils.data.DataLoader(cat_set, batch_size=3, shuffle=True,
#                                      num_workers=8)
# print(f'len titan {len(titan)}, len pie {len(pie)}, len nusc {len(nusc)}, len bdd {len(bdd)}')
i = 0

# for dataset in (titan, pie, nusc, bdd):
#   for d in dataset:
#       i+= 1
#       print(i)
#       print(f'len titan {len(titan)}, len pie {len(pie)}, len nusc {len(nusc)}, len bdd {len(bdd)}')
#       if d['obs_skeletons'].shape[0] != 2:
#          print('dataset', d['dataset_name'])

# loader = iter(loader)
# while True:
#     try:
#         d = next(loader)
#         print('dataset', d['dataset_name'])
#         print(d['hflip_flag'])
#         print(i)
#         print(f'len titan {len(titan)}, len pie {len(pie)}, len nusc {len(nusc)}, len bdd {len(bdd)}')
#     except:
#         print('dataset', d['dataset_name'])
#         print()
#         raise ValueError
#     i += 1

# oripath = '/home/y_feng/workspace6/datasets/JAAD/images/video_0082/00042.png'
# segroot = '/home/y_feng/workspace6/datasets/JAAD/seg_sam'
# segs = []
# for cls in os.listdir(segroot):
#     segpath = os.path.join(segroot, cls, '82/00042.pkl')
#     with open(segpath, 'rb') as f:
#         segs.append(pickle.load(f))
# print(segs[0].shape)
# colors = np.array([[[[255,0,0]]], [[[0,255,0]]], [[[0,0,255]]], [[[255,255,0]]]])
# img = np.zeros([1080, 1920, 3])
# for i in range(4):
#     color = colors[i]
#     seg = segs[i]
#     seg = np.expand_dims(seg, -1) * 1. * color

#     img += seg

# cv2.imwrite('./seg_sample2.png', img)

# param = torch.nn.Linear(2, 3).weight
# optimizer = torch.optim.Adam([param], lr=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
#                                                        T_max=10,
#                                                        verbose=True)
# warmer = warmup.LinearWarmup(optimizer, warmup_period=4)
# for i in range(5):
#     print(optimizer.state_dict()['param_groups'][0]['lr'])
#     optimizer.step()
#     with warmer.dampening():
#         scheduler.step()


pcpa = PCPA()
pedgraph = PedGraph()

print('PCPA')
for n, p in pcpa.named_parameters():
    if 'encoder' in n or 'proj' in n or'embedder' in n:
        continue
    print(n, p.size())

print('pedgraph')
for n, p in pedgraph.named_parameters():
    if 'encoder' in n or 'proj' in n or'embedder' in n:
        continue
    print(n, p.size())