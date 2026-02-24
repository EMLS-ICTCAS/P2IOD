import torch
import matplotlib.pyplot as plt
import os

class ValueBasedFuse():
    def __init__(self, topk_percent_ori=0.5, topk_percent_new=0.5):
        self.topk_percent_ori=topk_percent_ori
        self.topk_percent_new=topk_percent_new

    def merge(self, model_param_ori_task, model_param_new_task, model_param_init_task,vis_save_path=None):
        vector_ori_init_task = model_param_ori_task - model_param_init_task
        vector_new_ori_task = model_param_new_task - model_param_ori_task
        vector_new_init_task = model_param_new_task - model_param_init_task

        amplitude_ori_task = torch.abs(vector_ori_init_task)
        amplitude_new_task = torch.abs(vector_new_ori_task)

        mask_topk_percent_ori_task = self.select_topk_percent(amplitude_ori_task, self.topk_percent_ori)
        mask_topk_percent_new_task = self.select_topk_percent(amplitude_new_task, self.topk_percent_new)
        important_mask = mask_topk_percent_new_task & ~mask_topk_percent_ori_task

        not_important_mask = ~(mask_topk_percent_ori_task | mask_topk_percent_new_task)

        direction_vector_ori_task = torch.where(vector_ori_init_task > 0, 1, torch.where(vector_ori_init_task < 0, -1, 0))
        direction_vector_new_task = torch.where(vector_new_init_task > 0, 1, torch.where(vector_new_init_task < 0, -1, 0))
        direction_consist_mask = direction_vector_ori_task * direction_vector_new_task >= 0

        average_mask = not_important_mask & direction_consist_mask

        replace_mask = important_mask

        merge_model_param = model_param_ori_task + vector_new_ori_task*replace_mask.int()

        merge_model_param = torch.where(average_mask, (model_param_ori_task + model_param_new_task) / 2, merge_model_param)

        if vis_save_path is not None:
            self.visualize_prompt_parameter(model_param_new_task, os.path.join(vis_save_path,'param_new_task.png'))
            self.visualize_prompt_parameter(model_param_ori_task, os.path.join(vis_save_path,'param_ori_task.png'))
            self.visualize_prompt_parameter(model_param_init_task, os.path.join(vis_save_path,'param_init_task.png'))
            self.visualize_prompt_parameter(merge_model_param, os.path.join(vis_save_path,'merge_model_param.png'))

        return merge_model_param
    
    def select_topk_percent(self,tensor,topk_percent):
        threshold = torch.quantile(tensor, 1 - topk_percent)
        mask = (tensor >= threshold)
        return mask
    
    def visualize_prompt_parameter(self,prompt,save_path):
        data = prompt.cpu().numpy().flatten()

        plt.figure(figsize=(12, 6))

        plt.plot(data, marker='o', linestyle='-', markersize=3)

        plt.title('Tensor 可视化')
        plt.xlabel('索引')
        plt.ylabel('值')

        plt.grid(True)

        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    
