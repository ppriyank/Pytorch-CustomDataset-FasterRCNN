
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

from dataset import Dataset
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
parser.add_argument('--train-batch', default=32, type=int,
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
valid_anchors = valid_anchors(anchor_sizes,anchor_ratios , downscale , output_width=out_w , resized_width=width , output_height=out_h , resized_height=height)


dataset_train =  Dataset(data_folder=args.dataset, anchor_sizes = anchor_sizes, anchor_ratios = anchor_ratios, valid_anchors=valid_anchors, rev_label_map=config.rev_label_map,  split='TRAIN', image_resize_size= (height, width),  debug= False)
dataset_test =  Dataset(data_folder=args.dataset, anchor_sizes = anchor_sizes, anchor_ratios = anchor_ratios, valid_anchors=valid_anchors, rev_label_map=config.rev_label_map,  split='TEST', image_resize_size= (height, width),  debug= False)

trainloader = DataLoader(
    dataset_train, shuffle=True,  collate_fn=collate_fn, 
    batch_size=args.train_batch, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)


c= next(trainloader)
Y = c[3]

cls = Y[0][0]
regr = Y[1][0]


trans = transforms.ToPILImage()
img = trans(c[0]).convert("RGB")
box = c[1]
labels = c[2]
debug_num_pos = c[-1]   
verify2(img, box, labels="GT", config= config , color='#e6194b' , name="ground_truth")
n_anchratios = len(anchor_ratios)




c = config()
rpm = RPM(anchor_sizes , anchor_ratios, valid_anchors, c.rev_label_map)



lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

num_rois = 4 # Number of RoIs to process at once.
random.seed(1)


std_scaling = 4




input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)

