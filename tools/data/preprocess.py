import cv2
import numpy as np
import os
import pickle
import json
from ..utils import makedir


# bdd100k procedure: get vid_id2nm --> crop images --> get skeletons & segmentation maps

def crop_img(img, bbox, resize_mode, target_size=(224, 224)):
    l, t, r, b = list(map(int, bbox))
    cropped = img[t:b, l:r]
    if resize_mode == 'ori':
        resized = cropped
    elif resize_mode == 'resized':
        resized = cv2.resize(cropped, target_size)
    elif resize_mode == 'even_padded':
        h = b-t
        w = r-l
        if  float(w) / h > float(target_size[0]) / target_size[1]:
            ratio = float(target_size[0]) / w
        else:
            ratio = float(target_size[1]) / h
        new_size = (int(w*ratio), int(h*ratio))
        # print(cropped.shape, l, t, r, b, new_size)
        for l in new_size:
            if l == 0:
                return None
        cropped = cv2.resize(cropped, new_size)
        w_pad = target_size[0] - new_size[0]
        h_pad = target_size[1] - new_size[1]
        l_pad = w_pad // 2
        r_pad = w_pad - l_pad
        t_pad = h_pad // 2
        b_pad = h_pad - t_pad
        resized = cv2.copyMakeBorder(cropped,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
        assert (resized.shape[1], resized.shape[0]) == target_size
    
    return resized

def crop_ctx(img, bbox, mode, target_size=(224, 224)):
    ori_H, ori_W = img.shape[:2]
    l, t, r, b = list(map(int, bbox))
    # crop local context
    x = (l+r) // 2
    y = (t+b) // 2
    h = b-t
    w = r-l
    crop_h = h*2
    crop_w = h*2
    crop_l = max(x-h, 0)
    crop_r = min(x+h, ori_W)
    crop_t = max(y-h, 0)
    crop_b = min(y+h, ori_W)
    if mode == 'local':
        # mask target pedestrian
        rect = np.array([[l, t], [r, t], [r, b], [l, b]])
        masked = cv2.fillConvexPoly(img, rect, (127, 127, 127))
        cropped = masked[crop_t:crop_b, crop_l:crop_r]
    elif mode == 'ori_local':
        cropped = img[crop_t:crop_b, crop_l:crop_r]
    l_pad = max(h-x, 0)
    r_pad = max(x+h-ori_W, 0)
    t_pad = max(h-y, 0)
    b_pad = max(y+h-ori_H, 0)
    cropped = cv2.copyMakeBorder(cropped, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    assert cropped.shape[0] == crop_h and cropped.shape[1] == crop_w, (cropped.shape, (crop_h, crop_w))
    resized = cv2.resize(cropped, target_size)

    return resized


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

def crop_img_ctx_bdd100k(data_root='/home/y_feng/workspace6/datasets/BDD100k/bdd100k',
                     sub_set='train_val',
                     resize_mode='even_padded',
                     ctx_format='ori_local',
                     img_size=(224, 224)):
    vid_nm2id_path = os.path.join(data_root, 'extra', 'vid_nm2id.pkl')
    with open(vid_nm2id_path, 'rb') as f:
        vid_nm2id = pickle.load(f)
    img_root = os.path.join(data_root, 'images', 'track')
    label_root = os.path.join(data_root, 'labels', 'box_track_20')
    crop_root = os.path.join(data_root, 
                             'extra', 
                             'cropped_images', 
                             resize_mode,
                             f'{img_size[1]}w_by_{img_size[0]}h')
    makedir(crop_root)
    if ctx_format != '':
        ctx_root = os.path.join(data_root, 
                                'extra', 
                                'context', 
                                ctx_format,
                                f'{img_size[1]}w_by_{img_size[0]}h')
        makedir(ctx_root)
    for _subset in sub_set.split('_'):
        label_dir = os.path.joni(label_root, _subset)
        # traverse video
        for lfnm in os.listdir(label_dir):
            l_path = os.path.join(label_dir, lfnm)
            with open(l_path) as f:
                vid_l = json.load(f)  # list
            vid_nm = lfnm.replace('.json', '')
            vid_id = vid_nm2id[vid_nm]
            # traverse label
            for img_l in vid_l:
                img_nm = img_l['name']
                img_id = img_nm.split('-')[-1].replace('.jpg', '')
                img_id_int = int(img_id)
                for l in img_l['labels']:
                    cls = l['category']
                    oid = l['id']
                    oid_int = int(oid)
                    cls_k = 'ped' if cls in ('other person', 'pedestrian', 'rider') else 'veh'
                    ltrb = [l['bbox_2d']['x1'],
                            l['bbox_2d']['y1'],
                            l['bbox_2d']['x2'],
                            l['bbox_2d']['y2']]
                    img = cv2.imread(os.path.join(img_root, 
                                                  _subset,
                                                  vid_nm,
                                                  img_nm))
                    # save cropped images
                    tgt_crop_dir = os.path.join(crop_root,
                                           cls_k,
                                           str(oid_int),
                                           )
                    makedir(tgt_crop_dir)
                    tgt_crop_path = os.path.join(tgt_crop_dir,
                                                 str(img_id_int)+'.png')
                    cropped = crop_img(img,
                                       ltrb,
                                       resize_mode,
                                       img_size)
                    cv2.imwrite(tgt_crop_path, cropped)
                    # save ctx
                    if ctx_format != '':
                        tgt_ctx_dir = os.path.join(ctx_root,
                                                    cls_k,
                                                    str(oid_int),
                                                    )
                        makedir(tgt_ctx_dir)
                        tgt_ctx_path = os.path.join(tgt_ctx_dir,
                                                    str(img_id_int)+'.png')
                        cropped = crop_ctx(img,
                                        ltrb,
                                        ctx_format,
                                        img_size)
                        cv2.imwrite(tgt_ctx_path, cropped)

def get_skeleton_bdd100k()
    


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
    vid_nm2id_path = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/extra/vid_nm2id.pkl'
    with open(vid_nm2id_path, 'rb') as f:
        vid_nm2id = pickle.load(f)
    label_root = '/home/y_feng/workspace6/datasets/BDD100k/bdd100k/labels/box_track_20'

    vidid2oid = {}
    oids_all = set()
    num_obj = 0
    for subset in os.listdir(label_root):
        set_dir = os.path.join(label_root, subset)
        for fnm in os.listdir(set_dir):
            vidid = fnm.replace('.json', '')
            with open(os.path.join(set_dir, fnm)) as f:
                vid_content = json.load(f)
            cls2oid = {}
            for frame in vid_content:
                for label in frame['labels']:
                    cls = label['category']
                    if cls not in cls2oid:
                        cls2oid[cls] = set()
                    cls2oid[cls].add(label['id'])
            for cls in cls2oid:
                oids_all = oids_all.union(cls2oid[cls])
                num_obj += len(cls2oid[cls])
    print(len(oids_all), num_obj)