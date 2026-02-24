# PP-IOD
## results log
Due to file size limitations, we have only uploaded the training log files (stats.txt) for each incremental task, but not the model checkpoints.

Training log files are in "pp_iod_results" folder. 

## Installation
The code has been tested on Cuda 11.6 and Pytorch 1.13.1
Follow the steps below to install
```bash
conda create -n pp-iod python=3.9 -y
conda activate pp-iod
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install lightning==2.1.3
pip install pytorch-lightning==2.1.3
pip install mmengine==0.7.3
pip install transformers==4.37.2
pip install scipy
pip install matplotlib
pip install pycocotools
pip install timm
pip install numpy==1.24.4
```
## Data Preparation

Please put VOC2007 datasets to the ./dataset directory.
```
pp-iod
└── datasets
    └── VOC2007
        ├── Annotations
        ├── ImageSets
        └── JPEGImages    
```

First, run the following program to convert the VOC dataset to COCO format.
```python 
python datasets/pascal_voc_to_MS_COCO.py datasets --out-dir datasets
```

Next, run the following program to split the VOC and COCO datasets into incremental datasets.
```python 
# split voc 10+10 dataset
python datasets/split_incremental_task.py --n_tasks 2 --train_ann datasets/voc07/annotations/instances_train.json --test_ann datasets/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_10_10

# split voc 15+5 dataset
python datasets/split_incremental_task.py --n_tasks 2 --train_ann datasets/voc07/annotations/instances_train.json --test_ann datasets/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_15_5

# split voc 19+1 dataset
python datasets/split_incremental_task.py --n_tasks 2 --train_ann datasets/voc07/annotations/instances_train.json --test_ann datasets/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_19_1

# split voc 10+5+5 dataset
python datasets/split_incremental_task.py --n_tasks 3 --train_ann datasets/voc07/annotations/instances_train.json --test_ann datasets/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_10_5_5

# split voc 5+5+5+5 dataset
python datasets/split_incremental_task.py --n_tasks 4 --train_ann datasets/voc07/annotations/instances_train.json --test_ann datasets/voc07/annotations/instances_val.json --output_dir split_dataset --split_task voc_5_5_5_5

```

## Pretrain Model preparation
Please download the [COCO](https://huggingface.co/SenseTime/deformable-detr/tree/main) and [resnet50](https://huggingface.co/timm/resnet50.a1_in1k/tree/main) from Hugging Face, and place the model and configuration files into the "./coco_pretrained_Deformable_DETR", "./resnet50.a1_in1k" folders respectively.

Folder structure:
```text
PP-IOD/
├── coco_pretrained_Deformable_DETR/
│   ├── config.json
│   ├── gitattributes
│   ├── model.safetensors
│   ├── preprocessor_config
│   ├── pytorch_model.bin
│   └── README.md
└── resnet50.a1_in1k
    └── pytorch_model.bin
```


## Training MD-DETR

We provide training code for both single-step and multi-step scenarios.
```bash
# train voc in 10+10 setting with coco pretrained model
$ bash train_voc_10_10_coco_pretrain.sh

# train voc in 15+5 setting with coco pretrained model
$ bash train_voc_15_5_coco_pretrain.sh

# train voc in 19+1 setting with coco pretrained model
# It is worth noting that since the second task contains only one category and has very limited data, Deformable-DETR may suffer from overfitting during training. Therefore, compared to other tasks, we additionally adjusted the focal loss parameters (alpha=0.5 and gamma=3.0) of the detector to help mitigate the overfitting issue.
$ bash train_voc_19_1_coco_pretrain.sh

# train voc in 10+5+5 setting with coco pretrained model
$ bash train_voc_10_5_5_coco_pretrain.sh

# train voc in 5+5+5+5 setting with coco pretrained model
$ bash train_voc_5_5_5_5_coco_pretrain.sh

In training code, it has validation programs that provides detailed accuracy logs for each incremental tasks. If you would like to verify the accuracy of a specific model, you can run the script below.

## Evaluate checkpoint
# Please modify the model path "test_model_path"; the task-related parameters "task_id" and "split_task" in the script.
$ bash test_voc.sh
```

