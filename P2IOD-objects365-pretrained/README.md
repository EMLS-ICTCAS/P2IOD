# PP-IOD
## results log
Due to file size limitations, we have only uploaded the training log files (stats.txt) for each incremental task, but not the model checkpoints.

Training log files are in "pp_iod_results" folder. 

## Install
We implement PP-IOD using [MMDetection V2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3) and [MMCV V1.7.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.7.0).
We test our models under ```python=3.9,pytorch=1.13.1,cuda=11.6```. Other versions may not be compatible. 

Follow the steps below to install

```bash
conda create -n pp-iod-objects365-pretrained python=3.9 -y
conda activate pp-iod-objects365-pretrained
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install mmcv-full==1.7.0
pip install mmengine==0.7.3
pip install terminaltables
pip install six
pip install pycocotools
pip install fairscale
pip install fvcore
pip install scipy
pip install timm
pip install einops
pip install matplotlib
pip install yapf==0.40.1
pip install numpy==1.24.4
pip install setuptools==50.3.2
pip install -v -e .
```

## Data Preparation

Please put COCO and VOC2007 datasets to the ./data directory.
```
pp-iod
└── data
    ├── coco
    │   ├── annotations
    │   │      ├── instances_train2017.json
    │   │      └── instances_val2017.json
    │   ├── train2017
    │   └── val2017
    │
    └── VOC2007
        ├── Annotations
        ├── ImageSets
        └── JPEGImages    
```
First, run the following program to convert the VOC dataset to COCO format.
```python 
python data/pascal_voc_to_MS_COCO.py data --out-dir data
```

Next, run the following program to split the VOC and COCO datasets into incremental datasets.
```python 
# split voc 10+10 dataset
python data/split_incremental_task.py --n_tasks 2 --train_ann data/voc07/annotations/instances_train.json --test_ann data/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_10_10

# split voc 15+5 dataset
python data/split_incremental_task.py --n_tasks 2 --train_ann data/voc07/annotations/instances_train.json --test_ann data/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_15_5

# split voc 19+1 dataset
python data/split_incremental_task.py --n_tasks 2 --train_ann data/voc07/annotations/instances_train.json --test_ann data/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_19_1

# split voc 10+5+5 dataset
python data/split_incremental_task.py --n_tasks 3 --train_ann data/voc07/annotations/instances_train.json --test_ann data/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_10_5_5

# split voc 5+5+5+5 dataset
python data/split_incremental_task.py --n_tasks 4 --train_ann data/voc07/annotations/instances_train.json --test_ann data/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_5_5_5_5

# split coco 40+20+20 dataset
python data/split_incremental_task.py --n_tasks 3 --train_ann data/coco/annotations/instances_train2017.json --test_ann data/coco/annotations/instances_val2017.json --output_dir split_dataset --split_task coco_40_20_20

# split coco 40+10+10+10+10 dataset
python data/split_incremental_task.py --n_tasks 5 --train_ann data/coco/annotations/instances_train2017.json --test_ann data/coco/annotations/instances_val2017.json --output_dir split_dataset --split_task coco_40_10_10_10_10
```

## Pretrain Model preparation
Please download the [Objects365](https://huggingface.co/zongzhuofan/co-detr-vit-large-objects365),  from Hugging Face, and place the model into the "./Objects365_pretrained_codetr" folder.

Folder structure:
```text
PP-IOD-objects365-pretrained/
└── Objects365_pretrained_codetr/
    └── pytorch_model.bin

```


## Training MD-DETR

We provide training code for both single-step and multi-step scenarios.
```bash
# train voc in 10+10 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_voc_10_10_setting.sh

# train voc in 15+5 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_voc_15_5_setting.sh

# train voc in 19+1 setting with objects365 pretrained model
# It is worth noting that since the second task contains only one category and has very limited data, Detector may suffer from overfitting during training. Therefore, compared to other settings, we additionally adjusted the Quality Focal Loss parameters (beta=3.0) of the detector to help mitigate the overfitting issue.
bash train_objects365_pretrained_incre_voc_19_1_setting.sh

# train voc in 10+5+5 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_voc_10_5_5_setting.sh

# train voc in 5+5+5+5 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_voc_5_5_5_5_setting_train.sh

# train coco in 40+40 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_coco_40_40_setting.sh

# train coco in 50+30 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_coco_50_30_setting.sh

# train coco in 60+20 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_coco_60_20_setting.sh

# train coco in 70+10 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_coco_70_10_setting.sh

# train coco in 40+20+20 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_coco_40_20_20_setting.sh

# train coco in 40+10+10+10+10 setting with objects365 pretrained model
bash train_objects365_pretrained_incre_coco_40_10_10_10_10_setting.sh


```
