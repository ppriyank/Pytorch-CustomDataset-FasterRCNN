
import os
import sys
import argparse
import numpy as np
import math 

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


import model
from tools import * 

from dataset import Dataset , collate_fn
import torchvision.transforms as transforms

class Config(object):
    """docstring for config"""
    def __init__(self):
        super(Config, self).__init__()
        self.voc_labels = ('laptop', 'person', 'lights', 'drinks' , 'projector')
        self.label_map = {k: v for v, k in enumerate(self.voc_labels)}
        self.label_map['bg'] = len(self.label_map)
        self.rev_label_map = {v: k for k, v in self.label_map.items()}  # Inverse mapping
        # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        self.distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']



config = Config()


parser = argparse.ArgumentParser(description='Faster RCNN (Custom Dataset)')
parser.add_argument('--train-batch', default=1, type=int,
                    help="train batch size")

parser.add_argument('--workers', default=8, type=int,
                    help="# of workers")
parser.add_argument('--seed', default=1, type=int,
                    help="# seed")
parser.add_argument('--gpu-devices', default="0,1", type=str,
                    help="# of gpu devices")


parser.add_argument('-d', '--dataset', type=str, default='./',
                    help="path of the datatset...")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")

parser.add_argument('--height', type=int, default=800,
                    help="height of an image (default: 800)")
parser.add_argument('--width', type=int, default=600,
                    help="width of an image (default: 600)")

parser.add_argument('--anchor-sizes', default=None, type=list)
parser.add_argument('--anchor-ratio', default=[1,0.5,2], type=list)

parser.add_argument('--rpn-min-overlap', default=0.3, type=float,
                    help="min overlap")
parser.add_argument('--rpn-max-overlap', default=0.7, type=float,
                    help="max overlap with ground truth ")

parser.add_argument('--std_scaling', default=4.0, type=float,
                    help="scalling factor for regression ")



args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# args.gpu_devices = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()
pin_memory = True if use_gpu else False
cudnn.benchmark = True

height = args.height
width = args.width

# height = 3024
# width = 4032

height = args.height
width = args.width
std_scaling = args.std_scaling

out_h , out_w = base_size_calculator (height  , width)

# see convention figure 
downscale = max( 
        math.ceil(height / out_h) , 
        math.ceil(width / out_w)
        )


if args.anchor_sizes == None : 
    min_dim = min(height, width)
    index = math.floor(math.log(min_dim) /  math.log(2))
    args.anchor_sizes = [ 2 ** index , 2 ** (index-1) , 2 ** (index-2)]


anchor_ratios = args.anchor_ratio
anchor_sizes = args.anchor_sizes
num_anchors = len(anchor_ratios) * len(anchor_sizes)

valid_anchors = valid_anchors(anchor_sizes,anchor_ratios , downscale , output_width=out_w , resized_width=width , output_height=out_h , resized_height=height)

rpm = RPM(anchor_sizes , anchor_ratios, valid_anchors, config.rev_label_map, rpn_max_overlap=args.rpn_max_overlap , rpn_min_overlap= args.rpn_min_overlap)

dataset_train =  Dataset(data_folder=args.dataset, rpm=rpm, split='TRAIN', std_scaling=std_scaling, image_resize_size= (height, width),  debug= False)
dataset_test =  Dataset(data_folder=args.dataset, rpm=rpm, split='TEST', std_scaling=std_scaling, image_resize_size= (height, width),  debug= False)


train_loader = DataLoader(
    dataset_train, shuffle=True,  collate_fn=collate_fn, 
    batch_size=args.train_batch, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)


test_loader = DataLoader(
    dataset_test, shuffle=True,  collate_fn=collate_fn, 
    batch_size=1, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)


# sanity check 
# list(train_loader)

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

num_rois = 4 # Number of RoIs to process at once.
random.seed(1)




base_learning_rate =  0.00035
params = []
for key, value in model.named_parameters():
    if not value.requires_grad:
        continue
    lr = base_learning_rate
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]



optimizer_model = torch.optim.Adam(model_params)
optimizer_classifier = torch.optim.Adam(classifier_params)


model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')








         

