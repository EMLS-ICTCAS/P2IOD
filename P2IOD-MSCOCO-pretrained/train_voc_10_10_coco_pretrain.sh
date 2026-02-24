#!/bin/bash
# export CUDA_VISIBLE_DEVICES=2

use_prompts=1
plen=10
mlp_hidden_dim=64
exp_name="results/ppiod_for_voc_10_10_coco_pretrained_detector"
split_task="voc_10_10"

repo_name="./coco_pretrained_Deformable_DETR"
tr_dir="datasets/voc07/train"
val_dir="datasets/voc07/val"
task_ann_dir="split_dataset/"${split_task}

freeze='backbone,encoder,decoder,level_embed,input_proj,query_position_embeddings,reference_points'
new_params='class_embed,prompts'

EXP_DIR=./${exp_name}

echo "Training ... "${exp_name}
python train.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} --repo_name=${repo_name} \
    --n_gpus 1 --batch_size 2 --epochs 100 --lr_drop 80 --lr 1e-4 --lr_old 1e-5 --n_classes=21 --num_workers=2 --split_task=${split_task}\
    --use_prompts $use_prompts --prompt_len=$plen --mlp_hidden_dim=$mlp_hidden_dim --freeze=${freeze} --viz --new_params=${new_params} \
    --start_task=1 --n_tasks=2 --save_epochs=25 --eval_epochs=25  --bg_thres=0.65 --bg_thres_topk=5\
    --checkpoint_dir ${EXP_DIR}'/Task_1' --checkpoint_task1_name 'checkpoint99.pth' --checkpoint_other_task_name 'fused_model.pth' --resume=0 --parameterized_prompt_fusion --topk_percent_ori=0.5 \
    --topk_percent_new=0.5 --sparse_prompt --lambda_l1_norm=0.00001

