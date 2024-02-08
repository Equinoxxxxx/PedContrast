# Pytorch implementation for "Contrasting Disentangled Partial Observations for Pedestrian Action Prediction"

<img src="https://github.com/Equinoxxxxx/PedContrast/blob/master/fig1.png" width="500px">

## Prepare the datasets
Change the directory for datasets (```dataset_root```)
## Prepare the datasets
Download the datasets ([PIE](https://github.com/aras62/PIEPredict?tab=readme-ov-file#PIE_dataset), [JAAD](https://github.com/ykotseruba/JAAD), [TITAN](https://usa.honda-ri.com/titan), [nuScenes](https://www.nuscenes.org/nuscenes), [BDD100k](https://doc.bdd100k.com/download.html))  
Note: for PIE and JAAD, the original data are in .mp4 format, use the scripts from the official repo ([PIE](https://github.com/aras62/PIEPredict?tab=readme-ov-file#PIE_dataset), [JAAD](https://github.com/ykotseruba/JAAD)) to extract the frames; for BDD100k, we only use the MOT 2020 subset that contains continous bounding boxes.  

Extract the data in the following format:
```
[dataset_root]/
    PIE_dataset/
        annotations/
        PIE_clips/
        ...
    JAAD/
        annotations/
        JAAD_clips/
        ...
    TITAN/
        honda_titan_dataset/
            dataset/
                clip_1/
                clip_2/
                ...
    nusc/
        samples/
        v1.0-trainval/
        ...
    BDD100k/
        bdd100k/
            images/track/
                train/
                ...
            labels/box_track_20/
                train/
                ...
```
## Get the weights for Grounded SAM and HRNet
Download the weight files
```
[ckpt]
```

## Preprocess
Get cropped images, skeletons and segmentation maps
```
cd PedContrast
python -m tools.data.preprocess
```
