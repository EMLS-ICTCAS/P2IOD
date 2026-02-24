
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
import pdb
from torch.utils.data import DataLoader
from datetime import timedelta

import utils
import pytorch_lightning as pl
from datasets.coco_eval import CocoEvaluator
from engine import local_trainer, Evaluator
# from transformers import AutoImageProcessor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.coco_hug import CocoDetection
from datasets.split_incremental_task import task_info
from models.image_processing_deformable_detr import DeformableDetrImageProcessor 

import torch.distributed as dist

def get_args_parser():
    parser = argparse.ArgumentParser('MD-DETR', add_help=False)

    # Learning rate and optimizer parameters
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='Initial learning rate for training')
    parser.add_argument('--new_params', default='class_embed,prompts', type=str, 
                        help='New parameters to add to the model')
    parser.add_argument('--freeze', default='backbone,encoder,decoder', type=str, 
                        help='Parameters to freeze during training')
    parser.add_argument('--lr_old', default=1e-5, type=float, 
                        help='Learning rate for older layers')
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+', 
                        help='Names of backbone layers to apply learning rate to')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='Learning rate for backbone layers')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+', 
                        help='Layers to which a different learning rate multiplier is applied')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float, 
                        help='Multiplier for the learning rate on linear projection layers')

    # Batch, classes, and regularization parameters
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='Batch size for training')
    parser.add_argument('--n_classes', default=81, type=int, 
                        help='Number of object classes')
    parser.add_argument('--weight_decay', default=1e-4, type=float, 
                        help='Weight decay for optimizer regularization')

    # Epoch settings
    parser.add_argument('--epochs', default=6, type=int, 
                        help='Total number of training epochs')
    parser.add_argument('--eval_epochs', default=1, type=int, 
                        help='Number of epochs between evaluations')
    parser.add_argument('--print_freq', default=500, type=int, 
                        help='Frequency of printing training information (in steps)')


    # Learning rate schedule
    parser.add_argument('--lr_drop', default=5, type=int, 
                        help='Epoch to drop learning rate')
    parser.add_argument('--save_epochs', default=10, type=int, 
                        help='Interval of epochs between saving model checkpoints')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+', 
                        help='List of epochs to drop learning rate')
    
    # Gradient clipping
    parser.add_argument('--clip_max_norm', default=0.1, type=float, 
                        help='Maximum norm for gradient clipping')
    parser.add_argument('--sgd', action='store_true', 
                        help='Use SGD optimizer instead of AdamW')

    # Hardware and parallelism
    parser.add_argument('--n_gpus', default=1, type=int, 
                        help="Number of GPUs available for training")

    # Visualization
    parser.add_argument("--num_imgs_viz", type=int, default=30, 
                        help="Number of images for visualization during training")

    # Variants of Deformable DETR
    parser.add_argument('--repo_name', default="./deformable-detr", type=str, 
                        help='Repository name for the model')

    # Matcher cost coefficients
    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Coefficient for classification cost in matching")
    parser.add_argument('--set_cost_bbox', default=5, type=float, 
                        help="Coefficient for L1 box cost in matching")
    parser.add_argument('--set_cost_giou', default=2, type=float, 
                        help="Coefficient for GIoU box cost in matching")

    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="Coefficient for mask loss")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="Coefficient for dice loss")
    parser.add_argument('--cls_loss_coef', default=2, type=float, 
                        help="Coefficient for classification loss")
    parser.add_argument('--prompt_loss_coef', default=1, type=float, 
                        help="Coefficient for prompt loss")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="Coefficient for bounding box loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="Coefficient for GIoU loss")
    parser.add_argument('--focal_alpha', default=0.25, type=float, 
                        help="Alpha parameter for focal loss")
    parser.add_argument('--focal_gamma', default=2.0, type=float, 
                        help="Alpha parameter for focal loss")

    # Dataset and general training parameters
    parser.add_argument('--output_dir', default='./results/test', 
                        help='Directory to save outputs')
    parser.add_argument('--device', default='cuda', 
                        help='Device for training (default is CUDA)')
    parser.add_argument('--seed', default=42, type=int, 
                        help='Random seed')
    parser.add_argument('--resume', default=0, type=int, 
                        help='Resume training from a specific checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', 
                        help='Start training from a specific epoch')
    parser.add_argument('--eval', action='store_true', 
                        help='Run evaluation mode only')
    parser.add_argument('--viz', default=True, action='store_true', 
                        help='Run visualization only mode')
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='Number of workers for data loading')
    parser.add_argument('--cache_mode', default=False, action='store_true', 
                        help='Cache dataset in memory for faster training')

    # Continual learning setup
    parser.add_argument('--n_tasks', default=2, type=int, 
                        help='Number of tasks for continual learning setup')
    parser.add_argument('--start_task', default=1, type=int, 
                        help='Task to start training from in continual learning')
    parser.add_argument('--task_id', default=0, type=int, 
                        help='Task ID for continual learning')
    parser.add_argument('--reset_optim', default=1, type=int, 
                        help='Reset optimizer between tasks in continual learning')
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int, 
                        help='Number of classes introduced in previous tasks')
    parser.add_argument('--CUR_INTRODUCED_CLS', default=0, type=int, 
                        help='Number of new classes introduced in the current task')
    parser.add_argument('--mask_gradients', default=1, type=int, 
                        help='Flag to mask gradients during continual learning')

    # Checkpoint parameters
    parser.add_argument('--test_model_path', default='', type=str, 
                    help='Task ID for continual learning')   

    # Dataset paths
    parser.add_argument('--train_img_dir', default='', type=str, 
                        help='Training images directory')
    parser.add_argument('--test_img_dir', default='', type=str, 
                        help='Validation images directory')
    parser.add_argument('--task_ann_dir', default='', type=str, 
                        help='Directory for task annotations')
    parser.add_argument('--split_task', default='', type=str, 
                        help='Point to split training data for task setup')

    # Bounding box thresholds
    parser.add_argument('--bbox_thresh', default=0.3, type=float, 
                        help='Bounding box threshold for positive detections')
    parser.add_argument('--bg_thres', default=0.65, type=float, 
                        help='Threshold for considering a detection as background')
    parser.add_argument('--bg_thres_topk', default=5, type=int, 
                        help='Top-K background detections to consider')

    # Prompt memory-related parameters
    parser.add_argument("--use_prompts", type=int, default=1, 
                        help="Enable or disable use of prompt memory")
    parser.add_argument("--prompt_num", type=int, default=10, 
                        help="num of the prompt")
    parser.add_argument("--mlp_hidden_dim", type=int, default=10, 
                        help="hidden dim of paramerized prompt")

    # Parameterized Prompt Fusion
    parser.add_argument('--parameterized_prompt_fusion', default=True, action='store_true', 
                        help='Add sparse loss to regulate the sparse prompt')
    parser.add_argument('--topk_percent_ori', default=0.5, type=float, 
                        help='Fuse top-k percent parameter')
    parser.add_argument('--topk_percent_new', default=0.1, type=float, 
                        help='Fuse top-l percent parameter')
    
    # Sparse prompt
    parser.add_argument('--sparse_prompt', default=True, action='store_true', 
                        help='Add sparse loss to regulate the sparse prompt')
    parser.add_argument('--lambda_l1_norm', default=0.0001, type=float, 
                        help='Adjustable parameters for sparse loss')
    return parser

def main(args):

    # fix the seed for reproducibility
    seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    #Trainer = pl.Trainer(args)
    seed_everything(seed, workers=True)
    
    args.iou_types = ['bbox']
    out_dir_root = args.output_dir
    
    args.task_map, args.task_label2name =  task_info(args.split_task)
    args.task_label2name[args.n_classes-1] = "BG"

    if args.repo_name:
        processor = DeformableDetrImageProcessor.from_pretrained(args.repo_name)
    else:
        processor = DeformableDetrImageProcessor()

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}')
    logger = CSVLogger(save_dir=args.output_dir, name="lightning_logs")

    task_id = args.task_id

    args.output_dir = os.path.join(out_dir_root, 'Task_'+str(task_id))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_file = open(out_dir_root+'/Task_'+str(task_id)+'_log.out', 'a')
    print('Logging: args ', args, file=args.log_file)

    #args.switch = True
    args.task = str(task_id)

    #### automatic training schedule
    pyl_trainer = pl.Trainer(devices=list(range(args.n_gpus)), accelerator="gpu", max_epochs=args.epochs, 
                gradient_clip_val=0.1, accumulate_grad_batches=int(32/(args.n_gpus*args.batch_size)), \
                check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
                log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0)

    tst_ann = os.path.join(args.task_ann_dir,'test_task_'+str(task_id)+'.json')

    test_dataset = CocoDetection(img_folder=args.test_img_dir, 
                                ann_file=tst_ann, processor=processor)
    test_dataloader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    
    coco_evaluator = CocoEvaluator(test_dataset.coco, args.iou_types)

    local_evaluator = Evaluator(processor=processor, test_dataset=test_dataset,test_dataloader=test_dataloader,
                                coco_evaluator=coco_evaluator,args=args,task_label2name=args.task_label2name, task_name='cur')

    args.use_prompts=True

    trainer = local_trainer(train_loader=test_dataloader,val_loader=test_dataloader,
                                    test_dataset=test_dataset,args=args,local_evaluator=local_evaluator,task_id=task_id)


    trainer.resume(args.test_model_path)

    trainer.evaluator.local_eval = 1
    pyl_trainer.validate(trainer,test_dataloader)

    ####################### Evaluating on all known classes ###################################################
    known_task_ids = ''.join(str(i) for i in range(1,task_id+1))
    tst_ann_known = os.path.join(args.task_ann_dir,'test_task_'+str(known_task_ids)+'.json')
    test_dataset_known = CocoDetection(img_folder=args.test_img_dir, 
                                ann_file=tst_ann_known, processor=processor)
    test_dataloader_known = DataLoader(test_dataset_known, collate_fn=test_dataset_known.collate_fn, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    
    args.task = known_task_ids
    coco_evaluator = CocoEvaluator(test_dataset_known.coco, args.iou_types)
    local_evaluator = Evaluator(processor=processor, test_dataset=test_dataset_known,test_dataloader=test_dataloader_known,
                                coco_evaluator=coco_evaluator,args=args,task_label2name=args.task_label2name,
                                local_trainer=trainer, local_eval=1, task_name='all')
    PREV_INTRODUCED_CLS = args.task_map[task_id][1]
    CUR_INTRODUCED_CLS = args.task_map[task_id][2]

    seen_classes = PREV_INTRODUCED_CLS + CUR_INTRODUCED_CLS
    invalid_cls_logits = list(range(seen_classes, args.n_classes-1))
    local_evaluator.invalid_cls_logits = invalid_cls_logits

    trainer.evaluator = local_evaluator
    trainer.evaluator.model = trainer.model
    trainer.eval_mode = True
    pyl_trainer.validate(trainer,test_dataloader_known)
    ##########################################################################################################

    args.log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    out_dir = args.output_dir

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
