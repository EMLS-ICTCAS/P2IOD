#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

use_prompts=1
pnum=10
mlp_hidden_dim=64
exp_name="results/test"
split_task="voc_10_10"

repo_name="./coco_pretrained_Deformable_DETR"
tr_dir="datasets/voc07/train"
val_dir="datasets/voc07/val"
task_ann_dir="split_dataset/"${split_task}

test_model_path="results/ppiod_for_voc_10_10_coco_pretrained_detector/Task_2/checkpoint99.pth"
task_id=2

EXP_DIR=./${exp_name}

echo "Training ... "${exp_name}
python test.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} --repo_name=${repo_name} \
    --n_gpus 1 --batch_size 2 --n_classes=21 --task_id=${task_id} --num_workers=2 --split_task=${split_task}\
    --use_prompts $use_prompts --prompt_num=$pnum --mlp_hidden_dim=$mlp_hidden_dim --viz\
    --test_model_path=${test_model_path}