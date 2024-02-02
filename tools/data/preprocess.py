import cv2
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from ..utils import makedir
from .crop_images import crop_img_ctx_bdd100k
from .get_skeletons import get_skeleton_bdd100k
from .get_segmentation import segment_dataset
from config import dataset_root


# bdd100k procedure: get vid_id2nm --> crop images --> get skeletons & segmentation maps

def bdd100k_get_vidnm2vidid(data_root='/home/y_feng/workspace6/datasets/BDD100k/bdd100k',
                            sub_set='train_val'):
    vid_nms = []
    id2nm = {}
    nm2id = {}
    _subsets = sub_set.split('_')
    for _subset in _subsets:
        path = os.path.join(data_root, 'images', 'track', _subset)
        for vid in os.listdir(path):
            vid_nms.append(vid)
    _vid_nms = set(vid_nms)
    assert len(_vid_nms) == len(vid_nms)
    for i in range(len(vid_nms)):
        nm = vid_nms[i]
        id2nm[i] = nm
        nm2id[nm] = i
    with open(os.path.join(data_root, 'extra', 'vid_id2nm.pkl'), 'wb') as f:
        pickle.dump(id2nm, f)
    with open(os.path.join(data_root, 'extra', 'vid_nm2id.pkl'), 'wb') as f:
        pickle.dump(nm2id, f)
    return id2nm, nm2id


def prepare_data():
    # bdd100k
    # crop images
    # crop_img_ctx_bdd100k(img_size=(224, 224))
    # crop_img_ctx_bdd100k(ctx_format='', img_size=(384, 288))
    print('Get skeletons of bdd100k')
    get_skeleton_bdd100k(img_root=os.path.join(dataset_root, 'BDD100k/bdd100k/extra/cropped_images/even_padded/288w_by_384h/ped'),
                         tgt_root=os.path.join(dataset_root, 'BDD100k/bdd100k/extra/'))
    print('Get segmentation maps of bdd100k')
    for prompt in ('person', 'vehicle', 'road', 'traffic light'):
        print(prompt)
        segment_dataset(dataset_names='bdd100k',
                        prompt=prompt)

if __name__ == '__main__':
    # get video name to id
    # bdd100k_get_vidnm2vidid()
    # path = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/extra/vid_id2nm.pkl'
    # with open(path, 'rb') as f:
    #     a = pickle.load(f)
    # for k in a:
    #     print(a[k])
    # print('', len(a))
    
    # check obj id
    # vid_nm2id_path = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/extra/vid_nm2id.pkl'
    # with open(vid_nm2id_path, 'rb') as f:
    #     vid_nm2id = pickle.load(f)
    # label_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20'

    # vidid2oid = {}
    # oids_all = set()
    # num_obj = 0
    # for subset in os.listdir(label_root):
    #     set_dir = os.path.join(label_root, subset)
    #     for fnm in os.listdir(set_dir):
    #         vidid = fnm.replace('.json', '')
    #         with open(os.path.join(set_dir, fnm)) as f:
    #             vid_content = json.load(f)
    #         cls2oid = {}
    #         for frame in vid_content:
    #             for label in frame['labels']:
    #                 cls = label['category']
    #                 if cls not in cls2oid:
    #                     cls2oid[cls] = set()
    #                 cls2oid[cls].add(label['id'])
    #         for cls in cls2oid:
    #             oids_all = oids_all.union(cls2oid[cls])
    #             num_obj += len(cls2oid[cls])
    # print(len(oids_all), num_obj)
    prepare_data()