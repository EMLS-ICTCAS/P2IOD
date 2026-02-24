#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ngpu=8
portnum=29504

work_dir='results/co_dino_vit_ppiod_coco_50_30'
config_dir='projects/configs/co_dino_vit_ppiod_coco_50_30'

# train task1
# bash tools/dist_train.sh $config_dir'/co_dino_5scale_vit_large_coco_task1.py' $ngpu ${work_dir}'/co_dino_5scale_vit_large_coco_task1' $portnum
# train task2
bash tools/dist_train.sh $config_dir'/co_dino_5scale_vit_large_coco_task2.py' $ngpu ${work_dir}'/co_dino_5scale_vit_large_coco_task2' $portnum

python tools/value_based_fuse_consist_average.py \
        --init_prompts_path ${work_dir}'/co_dino_5scale_vit_large_coco_task1/init_prompt_mlp.pth' \
        --previous_model_path ${work_dir}'/co_dino_5scale_vit_large_coco_task1/latest.pth' \
        --current_model_path ${work_dir}'/co_dino_5scale_vit_large_coco_task2/latest.pth' \
        --current_model_config ${work_dir}'/co_dino_5scale_vit_large_coco_task2/co_dino_5scale_vit_large_coco_task2.py' \
        --output_checkpoint_dir ${work_dir}'/co_dino_5scale_vit_large_coco_task2/fused_model.pth' \
        --topk_percent_previous 0.3 \
        --topk_percent_current 0.3

bash tools/dist_test.sh ${work_dir}'/co_dino_5scale_vit_large_coco_task2/co_dino_5scale_vit_large_coco_task2.py' ${work_dir}'/co_dino_5scale_vit_large_coco_task2/fused_model.pth' ${work_dir}'/co_dino_5scale_vit_large_coco_task2' $ngpu $portnum &> ${work_dir}/co_dino_5scale_vit_large_coco_task2/test_after_fusion.txt
echo "所有脚本都已运行完毕。"