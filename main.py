import os
import pickle
import time
from turtle import resizemode
import argparse
import copy
import numpy as np
import pytorch_warmup as warmup

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tools.distributed_parallel import ddp_setup
torch.multiprocessing.set_sharing_strategy('file_system')

from tools.datasets.PIE_JAAD import PIEDataset
from tools.datasets.TITAN import TITAN_dataset
from tools.datasets.nuscenes_dataset import NuscDataset
from tools.datasets.bdd100k import BDD100kDataset

from models.PCPA import PCPA
from models.ped_graph23 import PedGraph

from tools.utils import makedir
from tools.log import create_logger
from tools.utils import draw_proto_info_curves, save_model, freeze, cls_weights, seed_all
from tools.plot import vis_weight_single_cls, draw_logits_histogram, draw_multi_task_curve, draw_curves2

from train_test import contrast_epoch, train_test_epoch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def get_args():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--ddp", default=False, type=bool)
    # data
    parser.add_argument('--pre_dataset_names', type=str, default='PIE')
    parser.add_argument('--train_dataset_names', type=str, default='TITAN')
    parser.add_argument('--test_dataset_names', type=str, default='TITAN_PIE')
    parser.add_argument('--small_set', type=float, default=0)
    parser.add_argument('--p_small_set', type=float, default=0)
    parser.add_argument('--test_small_set', type=float, default=0)
    parser.add_argument('--obs_len', type=int, default=4)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--obs_fps', type=int, default=2)
    parser.add_argument('--apply_tte', type=int, default=1)
    parser.add_argument('--test_apply_tte', type=int, default=1)
    parser.add_argument('--augment_mode', type=str, default='random_hflip')
    parser.add_argument('--img_norm_mode', type=str, default='torch')
    parser.add_argument('--color_order', type=str, default='BGR')
    parser.add_argument('--resize_mode', type=str, default='even_padded')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--p_overlap', type=float, default=0.5)
    parser.add_argument('--test_overlap', type=float, default=0.5)
    parser.add_argument('--dataloader_workers', type=int, default=8)
    parser.add_argument('--shuffle', type=int, default=1)

    # train
    parser.add_argument('--p_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--p_warm_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--p_batch_size', type=int, default=128)
    parser.add_argument('--test_every', type=int, default=2)
    parser.add_argument('--explain_every', type=int, default=10)
    parser.add_argument('--vis_every', type=int, default=2)
    parser.add_argument('--p_lr', type=float, default=0.)
    parser.add_argument('--p_backbone_lr', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--backbone_lr', type=float, default=0.0002)
    parser.add_argument('--scheduler', type=str, default='onecycle')
    parser.add_argument('--p_scheduler', type=str, default='onecycle')
    parser.add_argument('--p_onecycle_div_f', type=int, default=10)
    parser.add_argument('--p_batch_schedule', type=int, default=0)
    parser.add_argument('--lr_step_size', type=int, default=5)
    parser.add_argument('--lr_step_gamma', type=float, default=1.)
    parser.add_argument('--t_max', type=float, default=10)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--act_sets', type=str, default='cross')
    parser.add_argument('--key_metric', type=str, default='f1')
    parser.add_argument('--key_act_set', type=str, default='cross')

    # loss
    parser.add_argument('--loss_func', type=str, default='weighted_ce')
    parser.add_argument('--contrast_eff', type=float, default=0.)
    parser.add_argument('--logsig_thresh', type=float, default=100)
    parser.add_argument('--logsig_loss_eff', type=float, default=0.1)
    parser.add_argument('--logsig_loss', type=str, default='kl')

    # model
    parser.add_argument('--model_name', type=str, default='PCPA')
    parser.add_argument('--concept_mode', type=str, default='mlp_fuse')
    parser.add_argument('--pair_mode', type=str, default='pair_wise')
    parser.add_argument('--simi_func', type=str, default='dot_prod')
    parser.add_argument('--bridge_m', type=str, default='sk')
    parser.add_argument('--n_proto', type=int, default=50)
    parser.add_argument('--proj_dim', type=int, default=0)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--n_layer_proj', type=int, default=3)
    parser.add_argument('--proj_norm', type=str, default='ln')
    parser.add_argument('--proj_actv', type=str, default='leakyrelu')
    parser.add_argument('--uncertainty', type=str, default='none')
    parser.add_argument('--n_sampling', type=int, default=3)
    parser.add_argument('--n_mlp', type=int, default=1)

    # modality
    parser.add_argument('--modalities', type=str, default='img_sklt_ctx_traj_ego')
    # img settingf
    parser.add_argument('--img_format', type=str, default='')
    parser.add_argument('--img_backbone_name', type=str, default='R3D18')
    # sk setting
    parser.add_argument('--sklt_format', type=str, default='0-1coord')
    parser.add_argument('--sklt_backbone_name', type=str, default='poseC3D')
    # ctx setting
    parser.add_argument('--ctx_format', type=str, default='ori_local')
    parser.add_argument('--seg_cls', type=str, default='person,vehicles,roads,traffic_lights')
    parser.add_argument('--fuse_mode', type=str, default='transformer')
    parser.add_argument('--ctx_backbone_name', type=str, default='C3D_t4')
    # traj setting
    parser.add_argument('--traj_format', type=str, default='ltrb')
    parser.add_argument('--traj_backbone_name', type=str, default='lstm')
    # ego setting
    parser.add_argument('--ego_format', type=str, default='accel')
    parser.add_argument('--ego_backbone_name', type=str, default='lstm')

    args = parser.parse_args()

    return args

def main(rank, world_size, args):
    seed_all(42)
    # device
    local_rank = rank
    ddp = args.ddp and world_size > 1
    # data
    pre_dataset_names = args.pre_dataset_names.split('_')
    train_dataset_names = args.train_dataset_names.split('_')
    test_dataset_names = args.test_dataset_names.split('_')
    p_small_set = args.p_small_set
    small_set = args.small_set
    test_small_set = args.test_small_set
    obs_len = args.obs_len
    pred_len = args.pred_len
    obs_fps = args.obs_fps
    tte = None
    test_tte = None
    apply_tte = args.apply_tte
    test_apply_tte = args.test_apply_tte
    if apply_tte:
        tte = [0, int((obs_len+pred_len+1)/obs_fps*30)]  # before donwsample
    if test_apply_tte:
        test_tte = [0, int((obs_len+pred_len+1)/obs_fps*30)]  # before donwsample
    augment_mode = args.augment_mode
    img_norm_mode = args.img_norm_mode
    color_order = args.color_order
    resize_mode = args.resize_mode
    overlap = args.overlap
    p_overlap = args.p_overlap
    dataloader_workers = args.dataloader_workers
    shuffle = args.shuffle
    # train
    p_epochs = args.p_epochs
    epochs = args.epochs
    p_warm_step = args.p_warm_step
    batch_size = args.batch_size
    p_batch_size = args.p_batch_size
    test_every = args.test_every
    explain_every = args.explain_every
    vis_every = args.vis_every
    lr = args.lr
    backbone_lr = args.backbone_lr
    p_lr = args.p_lr
    p_backbone_lr = args.p_backbone_lr
    scheduler = args.scheduler
    p_scheduler = args.p_scheduler
    p_onecycle_div_f = args.p_onecycle_div_f
    p_batch_schedule = args.p_batch_schedule
    lr_step_size = args.lr_step_size
    lr_step_gamma = args.lr_step_gamma
    t_max = args.t_max
    optim = args.optim
    wd = args.weight_decay
    act_sets = args.act_sets.split('_')
    key_metric = args.key_metric
    key_act_set = args.key_act_set
    if len(act_sets) == 1:
        key_act_set = act_sets[0]
    assert key_act_set in act_sets + ['macro']

    # loss
    loss_func = args.loss_func
    contrast_eff = args.contrast_eff
    logsig_thresh = args.logsig_thresh
    logsig_loss_eff = args.logsig_loss_eff
    logsig_loss_func = args.logsig_loss
    # model
    model_name = args.model_name
    simi_func = args.simi_func
    pair_mode = args.pair_mode
    bridge_m = args.bridge_m
    n_proto = args.n_proto
    proj_dim = args.proj_dim
    pool = args.pool
    n_layer_proj = args.n_layer_proj
    proj_norm = args.proj_norm
    proj_actv = args.proj_actv
    uncertainty = args.uncertainty
    n_sampling = args.n_sampling
    n_mlp = args.n_mlp
    # modality
    modalities = args.modalities.split('_')
    img_format = args.img_format
    img_backbone_name = args.img_backbone_name
    sklt_format = args.sklt_format
    sk_backbone_name = args.sklt_backbone_name
    ctx_format = args.ctx_format
    seg_cls = args.seg_cls
    fuse_mode = args.fuse_mode
    ctx_backbone_name = args.ctx_backbone_name
    traj_format = args.traj_format
    traj_backbone_name = args.traj_backbone_name
    ego_format = args.ego_format
    ego_backbone_name = args.ego_backbone_name

    # conditioned config
    if model_name == 'PCPA':
        if 'JAAD' in test_dataset_names:
            modalities = ['sklt','ctx', 'traj']
        else:
            modalities = ['sklt','ctx', 'traj', 'ego']
        sklt_format = 'coord'
        if '0-1' in sklt_format:
            sklt_format = '0-1coord'
        ctx_format = 'ori_local'
    if model_name == 'ped_graph':
        if 'JAAD' in test_dataset_names:
            modalities = ['sklt','ctx']
        else:
            modalities = ['sklt','ctx', 'ego']
        sklt_format = 'coord'
        if '0-1' in sklt_format:
            sklt_format = '0-1coord'
        ctx_format = 'ped_graph'

    if 'R3D' in img_backbone_name or 'csn' in img_backbone_name\
        or 'R3D' in ctx_backbone_name or 'csn' in ctx_backbone_name:
        img_norm_mode = 'kinetics'
    if img_norm_mode in ('kinetics', '0.5', 'activitynet'):
        color_order = 'RGB'
    else:
        color_order = 'BGR'
    
    if uncertainty != 'gaussian':
        logsig_loss_eff = 0

    # # modality settings
    # m_settings = {m: {
    #     'data_format': locals()[m+'_format'],
    #     'backbone_name': locals()[m+'_backbone_name'],
    #     'pool': pool,
    #     'n_layer_proj': n_layer_proj,
    #     'norm': proj_norm,
    #     'proj_dim': proj_dim,
    #     'uncertainty': uncertainty,
    #     'n_sampling': n_sampling,
    # } for m in modalities}

    # create dirs
    exp_id = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    work_dir = '../work_dirs/exp/contrast'
    makedir(work_dir)
    model_type = model_name
    for m in modalities:
        model_type += '_' + m
    model_dir = os.path.join(work_dir, model_type, exp_id)
    print('Save dir of current exp: ', model_dir)
    makedir(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)
    plot_dir = os.path.join(model_dir, 'plot')
    makedir(plot_dir)
    pretrain_plot_dir = os.path.join(plot_dir, 'pretrain')
    makedir(pretrain_plot_dir)
    train_test_plot_dir = os.path.join(plot_dir, 'train_test')
    makedir(train_test_plot_dir)
    # logger
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    log('--------args----------')
    for k in list(vars(args).keys()):
        log(str(k)+': '+str(vars(args)[k]))
    log('--------args----------\n')
    args_dir = os.path.join(model_dir, 'args.pkl')
    with open(args_dir, 'wb') as f:
        pickle.dump(args, f)
    
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ddp:
        ddp_setup(local_rank, world_size=torch.cuda.device_count())
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device=torch.device("cuda", local_rank)
    
    # load the data
    log('----------------------------Load data-----------------------------')
    pre_datasets = {k:None for k in pre_dataset_names}
    train_datasets = {k:None for k in train_dataset_names}
    test_datasets = {k:None for k in test_dataset_names}
    val_datasets = {k:None for k in test_dataset_names}
    datasets = {
        'pre': pre_datasets,
        'train': train_datasets,
        'val': val_datasets,
        'test': test_datasets,
    }
    for subset in datasets:
        _subset = subset
        _overlap = overlap
        _small_set = test_small_set
        if subset == 'pre':
            _subset = 'train'
            _overlap = p_overlap
            _small_set = p_small_set
        elif subset == 'train':
            _small_set = small_set
        for name in datasets[subset]:
            if name == 'TITAN':
                cur_set = TITAN_dataset(sub_set='default_'+_subset, 
                                        norm_traj=False,
                                        img_norm_mode=img_norm_mode, color_order=color_order,
                                        obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                        obs_fps=obs_fps,
                                        recog_act=False,
                                        multi_label_cross=False, 
                                        use_atomic=False, 
                                        use_complex=False, 
                                        use_communicative=False, 
                                        use_transporting=False, 
                                        use_age=False,
                                        loss_weight='sklearn',
                                        small_set=_small_set,
                                        resize_mode=resize_mode, 
                                        modalities=modalities,
                                        img_format=img_format,
                                        sklt_format=sklt_format,
                                        ctx_format=ctx_format,
                                        traj_format=traj_format,
                                        ego_format=ego_format,
                                        augment_mode=augment_mode,
                                        )
            if name in ('PIE', 'JAAD'):
                cur_set = PIEDataset(dataset_name=name, seq_type='crossing',
                                     subset=_subset,
                                    obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                    obs_fps=obs_fps,
                                    do_balance=False, 
                                    bbox_size=(224, 224), 
                                    img_norm_mode=img_norm_mode, color_order=color_order,
                                    resize_mode=resize_mode,
                                    modalities=modalities,
                                    img_format=img_format,
                                    sklt_format=sklt_format,
                                    ctx_format=ctx_format,
                                    traj_format=traj_format,
                                    ego_format=ego_format,
                                    small_set=_small_set,
                                    tte=tte,
                                    recog_act=False,
                                    normalize_pos=False,
                                    augment_mode=augment_mode)
                if subset in ('test', 'val'):
                    cur_set.tte = test_tte
            if name == 'nuscenes':
                cur_set = NuscDataset(subset=_subset,
                                    obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                    obs_fps=obs_fps,
                                    small_set=_small_set,
                                    augment_mode=augment_mode,
                                    resize_mode=resize_mode,
                                    color_order=color_order, img_norm_mode=img_norm_mode,
                                    modalities=modalities,
                                    img_format=img_format,
                                    sklt_format=sklt_format,
                                    ctx_format=ctx_format,
                                    traj_format=traj_format,
                                    ego_format=ego_format
                                    )
            if name == 'bdd100k':
                cur_set = BDD100kDataset(subsets=_subset,
                                         obs_len=obs_len, pred_len=pred_len, overlap_ratio=_overlap, 
                                         obs_fps=obs_fps,
                                         color_order=color_order, img_norm_mode=img_norm_mode,
                                         small_set=_small_set,
                                         resize_mode=resize_mode,
                                         modalities=modalities,
                                         img_format=img_format,
                                         sklt_format=sklt_format,
                                         ctx_format=ctx_format,
                                         traj_format=traj_format,
                                         ego_format=ego_format,
                                         augment_mode=augment_mode
                                         )
            datasets[subset][name] = cur_set    
    for _sub in datasets:
        for nm in datasets[_sub]:
            if datasets[_sub][nm] is not None:
                log(f'{_sub} {nm} {len(datasets[_sub][nm])}')
    
    pre_cat_set = torch.utils.data.ConcatDataset([datasets['pre'][k] for k in datasets['pre']])

    train_sets = [datasets['train'][k] for k in datasets['train']]

    val_sets = [datasets['val'][k] for k in datasets['val']]
    test_sets = [datasets['test'][k] for k in datasets['test']]


    pre_loader = torch.utils.data.DataLoader(pre_cat_set, 
                                             batch_size=p_batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True)
    train_loaders = [torch.utils.data.DataLoader(cur_set, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle,
                                               num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in train_sets]
    val_loaders = [torch.utils.data.DataLoader(cur_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in val_sets]
    test_loaders = [torch.utils.data.DataLoader(cur_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle,
                                             num_workers=dataloader_workers,
                                             pin_memory=True
                                             ) for cur_set in test_sets]
    if ddp:
        raise NotImplementedError('ddp')
    
    # construct the model
    log('----------------------------Construct model-----------------------------')
    if model_name == 'PCPA':
        model = PCPA(modalities=modalities,
                     ctx_bb_nm=ctx_backbone_name,
                     proj_norm=proj_norm,
                     proj_actv=proj_actv,
                     pretrain=True,
                     act_sets=act_sets,
                     n_mlp=n_mlp,
                     proj_dim=proj_dim,
                     )
    elif model_name == 'ped_graph':
        model = PedGraph(modalities=modalities,
                         proj_norm=proj_norm,
                         proj_actv=proj_actv,
                         pretrain=True,
                         act_sets=act_sets,
                         n_mlp=n_mlp,
                         proj_dim=proj_dim,
                         )
    model = model.float().to(device)
    if ddp:
        raise NotImplementedError('ddp')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_parallel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)
    else:
        model_parallel = torch.nn.parallel.DataParallel(model)

    log('----------------------------Construct optimizer-----------------------------')
    # optimizer
    backbone_params, other_params = model.get_pretrain_params()
    p_opt_specs = [{'params': backbone_params, 'lr': p_backbone_lr},
                     {'params': other_params, 'lr':p_lr}]
    opt_specs = [{'params': backbone_params, 'lr': backbone_lr},
                     {'params': other_params, 'lr':lr}]
    
    if optim == 'sgd':
        p_optimizer = torch.optim.SGD(p_opt_specs, lr=p_backbone_lr, weight_decay=wd)
        optimizer = torch.optim.SGD(opt_specs, lr=lr, weight_decay=wd)
    elif optim == 'adam':
        p_optimizer = torch.optim.Adam(p_opt_specs, lr=p_backbone_lr, weight_decay=wd, eps=1e-5)
        optimizer = torch.optim.Adam(opt_specs, lr=lr, weight_decay=wd, eps=1e-5)
    elif optim == 'adamw':
        p_optimizer = torch.optim.AdamW(p_opt_specs, lr=p_backbone_lr, weight_decay=wd, eps=1e-5)
        optimizer = torch.optim.AdamW(opt_specs, lr=lr, weight_decay=wd, eps=1e-5)
    else:
        raise NotImplementedError(optim)
    
    # learning rate scheduler
    if p_scheduler == 'step':
        p_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=p_optimizer, 
                                                         step_size=lr_step_size, 
                                                         gamma=lr_step_gamma)
    elif p_scheduler == 'cosine':
        p_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=p_optimizer, 
                                                                    T_max=t_max, 
                                                                    eta_min=0)
    elif p_scheduler == 'onecycle':
        if p_epochs == 0:
            p_lr_scheduler = None
        else:
            p_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=p_optimizer, 
                                                                max_lr=p_backbone_lr*p_onecycle_div_f,
                                                                epochs=p_epochs,
                                                                steps_per_epoch=len(pre_loader),
                                                                div_factor=p_onecycle_div_f,
                                                                )
    else:
        raise NotImplementedError(p_scheduler)
    p_warmer = warmup.LinearWarmup(p_optimizer, 
                                   warmup_period=p_warm_step)
    if scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                       step_size=lr_step_size, 
                                                       gamma=lr_step_gamma)
    elif scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                                  T_max=t_max, 
                                                                  eta_min=0)
    elif scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                             max_lr=lr*10,
                                                             epochs=epochs,
                                                             steps_per_epoch=sum([len(l) for l in train_loaders]),
                                                             div_factor=10,
                                                             )
    else:
        raise NotImplementedError(scheduler)

    # train the model
    log('----------------------------Start training-----------------------------')
    curve_dict_dataset = {
        'train':{
            'acc':[],
            'auc':[],
            'f1':[],
            'map':[],
            'contrast_loss':[],
            'logsig_loss':[]
        },
        'val':{
            'acc':[],
            'auc':[],
            'f1':[],
            'map':[],
        },
        'test':{
            'acc':[],
            'auc':[],
            'f1':[],
            'map':[],
        }
    }
    curve_dict = {
        'TITAN': curve_dict_dataset,
        'PIE': copy.deepcopy(curve_dict_dataset),
        'JAAD': copy.deepcopy(curve_dict_dataset),
        'nuscenes': copy.deepcopy(curve_dict_dataset),
        'bdd100k': copy.deepcopy(curve_dict_dataset),
        # 'macro': copy.deepcopy(curve_dict_dataset),
    }
    pre_curve_dict = {
            'contrast_loss':[],
            'logsig_loss':[]
        }

    # pretrain
    for e in range(1, p_epochs+1):
        log(f'Pretrain {e} epoch')
        model_parallel.train()
        cur_lr = p_optimizer.state_dict()['param_groups'][0]['lr']
        log(f'lr: {cur_lr}')
        pre_res = contrast_epoch(model_parallel,
                                e,
                                pre_loader, 
                                optimizer=p_optimizer,
                                scheduler=p_lr_scheduler,
                                batch_schedule=p_batch_schedule,
                                warmer=p_warmer,
                                log=log, 
                                device=device,
                                modalities=modalities,
                                logsig_thresh=logsig_thresh,
                                logsig_loss_eff=logsig_loss_eff,
                                logsig_loss_func=logsig_loss_func,
                                exp_path=exp_id,
                                simi_func=simi_func,
                                pair_mode=pair_mode,
                                )
        for metric in pre_res:  # contrast loss, 
            pre_curve_dict[metric].append(pre_res[metric])
        if p_scheduler != 'onecycle' and not p_batch_schedule:
            with p_warmer.dampening():
                p_lr_scheduler.step()
        if e%vis_every == 0:
            draw_curves2(path=os.path.join(pretrain_plot_dir, 'contrast_and_logsig_loss.png'), 
                        val_lists=[pre_curve_dict['contrast_loss']],
                        labels=['contrast loss'],
                        colors=['r'],
                        vis_every=vis_every)
    # train/test
    best_test_res = {
        'acc': 0,
        'map': 0,
        'f1': 0,
        'auc': 0,
    }
    best_test_res_sep = {
        dataset_nm: {
        'acc': 0,
        'map': 0,
        'f1': 0,
        'auc': 0,
    } for dataset_nm in test_dataset_names}
    best_val_res = {
        'acc': 0,
        'map': 0,
        'f1': 0,
        'auc': 0,
    }
    best_val_res_sep = {
        dataset_nm: {
        'acc': 0,
        'map': 0,
        'f1': 0,
        'auc': 0,
    } for dataset_nm in test_dataset_names}
    best_e = -1
    for e in range(1, epochs+1):
        log(f'Fine tune {e} epoch')
        model_parallel.train()
        cur_lr = optimizer.state_dict()['param_groups'][1]['lr']
        log(f'cur lr: {cur_lr}')
        for loader in train_loaders:
            cur_dataset = loader.dataset.dataset_name
            log(cur_dataset)
            train_res = train_test_epoch(
                                        model_parallel, 
                                        e,
                                        loader,
                                        loss_func='weighted_ce',
                                        optimizer=optimizer,
                                        scheduler=lr_scheduler,
                                        log=log, 
                                        device=device,
                                        modalities=modalities,
                                        contrast_eff=contrast_eff,
                                        logsig_thresh=logsig_thresh,
                                        logsig_loss_eff=logsig_loss_eff,
                                        train_or_test='train',
                                        logsig_loss_func=logsig_loss_func,
                                        simi_func=simi_func,
                                        pair_mode=pair_mode
                                        )
            for metric in curve_dict[cur_dataset]['train']:
                if key_act_set == 'macro':
                    curve_dict[cur_dataset]['train'][metric].append(
                        sum([train_res[act_set][metric] for act_set in act_sets]) / len(act_sets)
                    )
                else:
                    curve_dict[cur_dataset]['train'][metric].append(
                        train_res[key_act_set][metric]
                    )
        if scheduler != 'onecycle':
            lr_scheduler.step()
        # validation and test
        if e%test_every == 0:
            model_parallel.eval()
            log(f'Val')
            for loader in val_loaders:
                cur_dataset = loader.dataset.dataset_name
                log(cur_dataset)
                val_res = train_test_epoch(
                                            model_parallel, 
                                            e,
                                            loader,
                                            loss_func='weighted_ce',
                                            optimizer=optimizer,
                                            log=log, 
                                            device=device,
                                            modalities=modalities,
                                            contrast_eff=contrast_eff,
                                            logsig_thresh=logsig_thresh,
                                            logsig_loss_eff=logsig_loss_eff,
                                            train_or_test='test',
                                            logsig_loss_func=logsig_loss_func,
                                            simi_func=simi_func,
                                            pair_mode=pair_mode
                                            )
                for metric in curve_dict[cur_dataset]['val']:
                    if key_act_set == 'macro':
                        curve_dict[cur_dataset]['val'][metric].append(
                            sum([val_res[act_set][metric] for act_set in act_sets]) / len(act_sets)
                        )
                    else:
                        curve_dict[cur_dataset]['val'][metric].append(
                            val_res[key_act_set][metric]
                        )
            log(f'Test')
            for loader in test_loaders:
                cur_dataset = loader.dataset.dataset_name
                log(cur_dataset)
                test_res = train_test_epoch(
                                            model_parallel, 
                                            e,
                                            loader,
                                            loss_func='weighted_ce',
                                            optimizer=optimizer,
                                            log=log, 
                                            device=device,
                                            modalities=modalities,
                                            contrast_eff=contrast_eff,
                                            logsig_thresh=logsig_thresh,
                                            logsig_loss_eff=logsig_loss_eff,
                                            train_or_test='test',
                                            logsig_loss_func=logsig_loss_func,
                                            simi_func=simi_func,
                                            pair_mode=pair_mode
                                            )
                # update result curves
                for metric in curve_dict[cur_dataset]['test']:
                    if key_act_set == 'macro':
                        curve_dict[cur_dataset]['test'][metric].append(
                            sum([test_res[act_set][metric] for act_set in act_sets]) / len(act_sets)
                        )
                    else:
                        curve_dict[cur_dataset]['test'][metric].append(
                            test_res[key_act_set][metric]
                        )
                    # draw curves
                    curve_list = [curve_dict[cur_dataset]['val'][metric],
                                  curve_dict[cur_dataset]['test'][metric]]
                    if cur_dataset in train_dataset_names:
                        curve_list.append(curve_dict[cur_dataset]['train'][metric])
                    # import pdb;pdb.set_trace()
                    draw_curves2(path=os.path.join(train_test_plot_dir, 
                                                   cur_dataset+'_'+metric+'.png'), 
                                val_lists=curve_list,
                                labels=['val', 'test', 'train'],
                                colors=['g', 'b', 'r'],
                                vis_every=vis_every)
            # save best results
            cur_key_res = sum([curve_dict[d]['val'][key_metric][-1] for d in test_dataset_names]) \
                            / len(test_dataset_names)
            if cur_key_res > best_val_res[key_metric]:
                for metric in best_val_res:
                    best_val_res[metric] = sum([curve_dict[d]['val'][metric][-1] for d in test_dataset_names]) \
                                            / len(test_dataset_names)
                    best_test_res[metric] = sum([curve_dict[d]['test'][metric][-1] for d in test_dataset_names]) \
                                            / len(test_dataset_names)
                    for dataset_nm in test_dataset_names:
                        best_test_res_sep[dataset_nm][metric] = curve_dict[dataset_nm]['test'][metric][-1]
                best_e = e
            log(f'Best val result:\n epoch {best_e}\n {best_val_res}\nBest test results:\n {best_test_res}')
            for metric in best_val_res:
                cur_metric_macro = sum([curve_dict[d]['test'][metric][-1] for d in test_dataset_names]) \
                                            / len(test_dataset_names)
                log(f' Cur metric macro test result: {metric} {cur_metric_macro}')
            if local_rank == 0 or not ddp:
                save_model(model=model, model_dir=ckpt_dir, 
                            model_name=str(e) + '_',
                            log=log)
    log(f'\nBest val result:\n epoch {best_e}\n {best_val_res}')
    log(f'\nBest test res: {best_test_res}')
    log(f'\nBest test res per dataset: {best_test_res_sep}')
    if p_epochs > 0:
        log(f'\nFinal contrastive loss:')
        log(pre_curve_dict['contrast_loss'][-1].cpu().numpy())
    logclose()
    with open(os.path.join(train_test_plot_dir, 'curve_dict.pkl'), 'wb') as f:
        pickle.dump(curve_dict, f)
    if ddp:
        raise NotImplementedError
        destroy_process_group()
    
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    args = get_args()
    if world_size > 1 and args.ddp:
        mp.spawn(main, args=(args),  nprocs=world_size)
    else:
        main(rank=0, world_size=world_size, args=args)