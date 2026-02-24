import torch
import matplotlib.pyplot as plt
import os
import argparse
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner.checkpoint import load_checkpoint
from projects import *
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--init_prompts_path', default='results/test/init_prompt_mlp.pth',help='init prompts path')
    parser.add_argument('--previous_model_path', default='results/co_dino_vit_iod_coco_40_40_unfreeze_cls_branches_and_prompts/co_dino_5scale_vit_large_coco_task1/epoch_1.pth',help='previous model path')
    parser.add_argument('--current_model_path', default='results/co_dino_vit_iod_coco_40_40_unfreeze_cls_branches_and_prompts/co_dino_5scale_vit_large_coco_task2/epoch_1.pth',help='current model path')
    parser.add_argument('--current_model_config', default='results/co_dino_vit_iod_coco_40_40_unfreeze_cls_branches_and_prompts/co_dino_5scale_vit_large_coco_task2/co_dino_5scale_vit_large_coco_task2.py',help='current model path')
    parser.add_argument('--output_checkpoint_dir', default='results/test/fused_model.pth',help='current model path')
    parser.add_argument('--topk_percent_previous', type=float, default=0.3,help='topk percent previous')
    parser.add_argument('--topk_percent_current', type=float, default=0.3,help='topk percent current')

    args = parser.parse_args()

    return args

class ValueBasedFuse():
    def __init__(self, topk_percent_previous=0.3, topk_percent_current=0.3):
        #topk增加的越多
        self.topk_percent_previous=topk_percent_previous
        self.topk_percent_current=topk_percent_current

    def merge(self, model_param_previous_task, model_param_new_task, model_param_init_task,vis_save_path=None):
        vector_previous_init_task = model_param_previous_task - model_param_init_task
        vector_new_previous_task = model_param_new_task - model_param_previous_task
        vector_new_init_task = model_param_new_task - model_param_init_task

        amplitude_previous_task = torch.abs(vector_previous_init_task)
        amplitude_new_task = torch.abs(vector_new_previous_task)

        # select mask是new_task幅值topk%中去除幅值topk%previous_task的位置
        mask_topk_percent_previous_task = self.select_topk_percent(amplitude_previous_task, self.topk_percent_previous)
        mask_topk_percent_current_task = self.select_topk_percent(amplitude_new_task, self.topk_percent_current)
        important_mask = mask_topk_percent_current_task & ~mask_topk_percent_previous_task

        not_important_mask = ~(mask_topk_percent_previous_task | mask_topk_percent_current_task)

        # direction_consist_mask是new_task和没有previous_task更新向量方向一致的位置
        direction_vector_previous_task = torch.where(vector_previous_init_task > 0, 1, torch.where(vector_previous_init_task < 0, -1, 0))
        direction_vector_new_task = torch.where(vector_new_init_task > 0, 1, torch.where(vector_new_init_task < 0, -1, 0))
        direction_consist_mask = direction_vector_previous_task * direction_vector_new_task >= 0

        average_mask = not_important_mask & direction_consist_mask

        # select mask和direction_consist_mask有一个满足要求就可以更新
        replace_mask = important_mask
        # merge_mask = select_mask

        merge_model_param = model_param_previous_task + vector_new_previous_task*replace_mask.int()

        merge_model_param = torch.where(average_mask, (model_param_previous_task + model_param_new_task) / 2, merge_model_param)

        if vis_save_path is not None:
            self.visualize_prompt_parameter(model_param_new_task, os.path.join(vis_save_path,'param_new_task.png'))
            self.visualize_prompt_parameter(model_param_previous_task, os.path.join(vis_save_path,'param_previous_task.png'))
            self.visualize_prompt_parameter(model_param_init_task, os.path.join(vis_save_path,'param_init_task.png'))
            self.visualize_prompt_parameter(merge_model_param, os.path.join(vis_save_path,'merge_model_param.png'))

        return merge_model_param
    
    def select_topk_percent(self,tensor,topk_percent):
        threshold = torch.quantile(tensor, 1 - topk_percent)
        mask = (tensor >= threshold)
        return mask
    
    def visualize_prompt_parameter(self,prompt,save_path):
        data = prompt.cpu().numpy().flatten()
        # 创建一个图形
        plt.figure(figsize=(12, 6))

        # 绘制折线图
        plt.plot(data, marker='o', linestyle='-', markersize=3)

        # 添加标题和标签
        plt.title('Tensor 可视化')
        plt.xlabel('索引')
        plt.ylabel('值')

        # 显示网格
        plt.grid(True)

        # 显示图形
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    

def main():
    args = parse_args()

    fuse_method = ValueBasedFuse(topk_percent_previous=args.topk_percent_previous, topk_percent_current=args.topk_percent_current)

    prompt_mlp_init = torch.load(args.init_prompts_path)
    model_current_ckpt = torch.load(args.current_model_path)

    cfg = Config.fromfile(args.current_model_config)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    load_checkpoint(model, args.previous_model_path)
    prompt_mlp_previous = model.query_head.transformer.prompts.get_prompt_memory_params()

    load_checkpoint(model, args.current_model_path)
    prompt_mlp_current = model.query_head.transformer.prompts.get_prompt_memory_params()

    merged_prompt_mlp = [
        fuse_method.merge(ori, new, init)
        for ori, new, init in zip(prompt_mlp_previous, prompt_mlp_current, prompt_mlp_init)
    ]

    model.query_head.transformer.prompts.set_memory_params(merged_prompt_mlp)

    new_ckpt = {
            'state_dict': model.state_dict(),
            'optimizer': model_current_ckpt['optimizer'],
            'meta': model_current_ckpt['meta'],
        }
    save_prompt_mlp_pth = args.output_checkpoint_dir
    torch.save(new_ckpt, save_prompt_mlp_pth)

if __name__ == '__main__':
    main()