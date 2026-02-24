#from coco_hug import create_task_json, task_info_coco
import argparse
from pathlib import Path
import json
import os
import numpy as np

def task_info(task_name):
    CLASS_NAMES_coco = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
    ] # n_classes = 80

    CLASS_NAMES_voc = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


    if task_name == 'coco_40_10_10_10_10':
        all_classes = CLASS_NAMES_coco

        task_map = {1:(all_classes[0:40],0,40),
                    2:(all_classes[40:50],40,10),
                    3:(all_classes[50:60],50,10),
                    4:(all_classes[60:70],60,10),
                    5:(all_classes[70:],70,10)
                    }
        task_label2name = {}
        for i,j in enumerate(all_classes):
            task_label2name[i] = j

    elif task_name == 'coco_40_20_20':
        all_classes = CLASS_NAMES_coco
        task_map = {1:(all_classes[0:40],0,40),
                    2:(all_classes[40:60],40,20),
                    3:(all_classes[60:],60,20)
                    }
        task_label2name = {}
        for i,j in enumerate(all_classes):
            task_label2name[i] = j

    elif task_name == 'voc_10_10':
        all_classes = CLASS_NAMES_voc
        T1_CLASS_NAMES = all_classes[:10]
        T2_CLASS_NAMES = all_classes[10:]

        all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
        task_map = {1: (T1_CLASS_NAMES, 0, 10), 2: (T2_CLASS_NAMES, 10, 10), }
        task_label2name = {}
        for i, j in enumerate(all_classes):
            task_label2name[i] = j
        return task_map, task_label2name
    
    elif task_name == 'voc_15_5':
        all_classes = CLASS_NAMES_voc
        T1_CLASS_NAMES = all_classes[:15]
        T2_CLASS_NAMES = all_classes[15:]

        all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
        task_map = {1: (T1_CLASS_NAMES, 0, 15), 2: (T2_CLASS_NAMES, 15, 5), }
        task_label2name = {}
        for i, j in enumerate(all_classes):
            task_label2name[i] = j
        return task_map, task_label2name

    elif task_name == 'voc_19_1':
        all_classes = CLASS_NAMES_voc
        T1_CLASS_NAMES = all_classes[:19]
        T2_CLASS_NAMES = all_classes[19:]

        all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
        task_map = {1: (T1_CLASS_NAMES, 0, 19), 2: (T2_CLASS_NAMES, 19, 1), }
        task_label2name = {}
        for i, j in enumerate(all_classes):
            task_label2name[i] = j
        return task_map, task_label2name

    elif task_name == 'voc_10_5_5':
        all_classes = CLASS_NAMES_voc
        T1_CLASS_NAMES = all_classes[:10]
        T2_CLASS_NAMES = all_classes[10:15]
        T3_CLASS_NAMES = all_classes[15:]

        all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES 
        task_map = {1: (T1_CLASS_NAMES, 0, 10), 2: (T2_CLASS_NAMES, 10, 5), 3: (T3_CLASS_NAMES, 15, 5)}
        task_label2name = {}
        for i, j in enumerate(all_classes):
            task_label2name[i] = j
        return task_map, task_label2name

    elif task_name == 'voc_5_5_5_5':
        all_classes = CLASS_NAMES_voc
        T1_CLASS_NAMES = all_classes[:5]
        T2_CLASS_NAMES = all_classes[5:10]
        T3_CLASS_NAMES = all_classes[10:15]
        T4_CLASS_NAMES = all_classes[15:]

        all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES 
        task_map = {1: (T1_CLASS_NAMES, 0, 5), 2: (T2_CLASS_NAMES, 5, 5), 3: (T3_CLASS_NAMES, 10, 5),
                    4: (T4_CLASS_NAMES, 15, 5) }
        task_label2name = {}
        for i, j in enumerate(all_classes):
            task_label2name[i] = j
        return task_map, task_label2name

    return task_map, task_label2name

def create_task_json(root_json, cat_names, set_type='train', offset=0, task_id=1, output_dir='', task_label2name=None):

    print ('Creating temp JSON for tasks ',task_id,' ...', set_type)
    temp_json = json.load(open(root_json, 'r'))

    id2id, name2id, flag = {}, {}, False

    if task_label2name:
        flag = True
        for i,j in task_label2name.items():
            name2id[j] = i

    cat_ids, keep_imgs = [], []

    for k,j in enumerate(temp_json['categories']):

        if not flag:
            name2id[j['name']] = j['id']

        if j['name'] in cat_names:
            cat_ids.append(j['id'])

    for j,i in enumerate(cat_names):
        id2id[name2id[i]] = offset+j
    
    data = {'images':[], 'annotations':[], 'categories':[], 
            'info':{},'licenses':[]}

    # count = 0
    #print ('total ', len(temp_json['annotations']))
    for i in temp_json['annotations']:
        # if count %100 ==0:
        #     print (count)
        # count+=1
        if i['category_id'] in cat_ids:
            temp = i
            #print (i, temp)
            temp['category_id'] = id2id[temp['category_id']]
            data['annotations'].append(temp)
            keep_imgs.append(i['image_id'])
    
    #print ('here')
            
    keep_imgs = set(keep_imgs)

    for i in temp_json['categories']:
        if i['id'] in cat_ids:
            temp = i
            temp['id'] = id2id[temp['id']]
            data['categories'].append(temp)
            #data['categories'].append(i)
    
    # data['info'] = temp_json['info']
    # data['licenses'] = temp_json['licenses']

    #count = 0
    print ('total images:', len(temp_json['images']), '  keeping:', len(keep_imgs))
    for i in temp_json['images']:
        if i['id'] in keep_imgs:
            data['images'].append(i)

    with open(os.path.join(output_dir,set_type+'_task_'+str(task_id)+'.json'),'w') as f:
        json.dump(data, f)
    

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--n_tasks', default=0, type=int)
    parser.add_argument('--train_ann', default="", type=str)
    parser.add_argument('--test_ann', default="", type=str)
    parser.add_argument('--output_dir', default="", type=str)
    parser.add_argument('--split_task',default='', type=str)
    
    return parser


def main(args):

    args.output_dir = os.path.join(args.output_dir,args.split_task)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    task_map, task_label2name =  task_info(args.split_task)

    for task_id in range(1,args.n_tasks+1):
        cur_task = task_map[task_id]

        create_task_json(root_json=args.train_ann,
                        cat_names=cur_task[0], offset=cur_task[1], set_type='train', output_dir=args.output_dir, task_id=task_id)

        create_task_json(root_json=args.test_ann,
                        cat_names=cur_task[0], offset=cur_task[1], set_type='test', output_dir=args.output_dir, task_id=task_id)

    if args.n_tasks == 2:
        create_task_json(root_json=args.test_ann,
                cat_names=task_map[1][0]+task_map[2][0], offset=0, set_type='test', output_dir=args.output_dir, task_id='12')

    for task_id in range(3,args.n_tasks+1):
        cur_task = task_map[task_id]
        known_task_ids = ''.join(str(i) for i in range(1,task_id))
        all_cat_names = []

        for i in range(1,task_id):
            all_cat_names.extend(task_map[i][0])
        
        create_task_json(root_json=args.test_ann,
                        cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)
    
    known_task_ids = ''.join(str(i) for i in range(1,task_id+1))
    all_cat_names = []

    for i in range(1,task_id+1):
        all_cat_names.extend(task_map[i][0])
    
    create_task_json(root_json=args.test_ann,
                    cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    print ('\n Done .... \n')