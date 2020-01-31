
import os
import sys
import argparse
import numpy as np
import math 
import pickle

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


from  model import Model_RPN , Classifier
from tools import * 
from utils import WarmupMultiStepLR 

from dataset import Dataset , collate_fn
import torchvision.transforms as transforms

from torch.autograd import Variable
from loss import rpn_loss_regr , rpn_loss_cls_fixed_num 




class Config(object):
    """docstring for config"""
    def __init__(self):
        super(Config, self).__init__()
        self.voc_labels = ('laptop', 'person', 'lights', 'drinks' , 'projector')
        self.voc_labels += ('bg',)
        self.label_map = {k: v for v, k in enumerate(self.voc_labels)}
        self.rev_label_map = {v: k for k, v in self.label_map.items()}  # Inverse mapping
        # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        self.distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']



config = Config()


parser = argparse.ArgumentParser(description='Faster RCNN (Custom Dataset)')
parser.add_argument('--train-batch', default=2, type=int,
                    help="train batch size")
parser.add_argument('--n-roi', type=int, default=20,
                    help="number of roi to train classifiers with")


parser.add_argument('--workers', default=8, type=int,
                    help="# of workers, keep it greater than 4")
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
random.seed(1)

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

# keep the number of workers greater than 4
train_loader = DataLoader(
    dataset_train, shuffle=True,  collate_fn=collate_fn, 
    batch_size=args.train_batch, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)


test_loader = DataLoader(
    dataset_test, shuffle=True,  collate_fn=collate_fn, 
    batch_size=1, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)




model_rpn = Model_RPN(num_anchors= len(anchor_sizes) * len(anchor_ratios) )
model_classifier = Classifier(num_classes=  len(config.voc_labels) )



weight_decay = 0.0005
base_learning_rate =  0.0035
params_class = []
for key, value in model_classifier.named_parameters():
    if not value.requires_grad:
        continue
    lr = base_learning_rate
    params_class += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


params_rpn = []
for key, value in model_rpn.named_parameters():
    if not value.requires_grad:
        continue
    lr = base_learning_rate
    params_rpn += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


optimizer_model_rpn = torch.optim.Adam(params_rpn)
optimizer_classifier = torch.optim.Adam(params_class)

scheduler_rpn = WarmupMultiStepLR(optimizer_model_rpn, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)
scheduler_class = WarmupMultiStepLR(optimizer_classifier, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)



all_possible_anchor_boxes = default_anchors(out_h=50, out_w=38, anchor_sizes=anchor_sizes , anchor_ratios=anchor_ratios , downscale=16)
all_possible_anchor_boxes_tensor = torch.tensor(all_possible_anchor_boxes)
# all_possible_anchor_boxes_tensor = all_possible_anchor_boxes_tensor.unsqueeze(0).repeat(args.train_batch, 1,1,1,1)
# torch.Size([4, 50, 38, 9])


for i,(image, boxes, labels , temp, num_pos) in enumerate(train_loader):
            break

y_is_box_label = temp[0]
y_rpn_regr = temp[1]


image = Variable(image)



for i in range(25) :
    base_x , cls_k , reg_k = model_rpn(image)
    l1 = rpn_loss_regr(y_true=y_rpn_regr, y_pred=reg_k , y_is_box_label=y_is_box_label)
    l2 = rpn_loss_cls_fixed_num(y_pred = cls_k , y_is_box_label= y_is_box_label)
    loss = l1 + l2 
    print(loss.item())
    optimizer_model_rpn.zero_grad()
    loss.backward()
    optimizer_model_rpn.step()


rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []


with torch.no_grad():
    base_x , cls_k , reg_k = model_rpn(image)


img_data = {}
for b in range(args.train_batch):
    with torch.no_grad():
        # Convert rpn layer to roi bboxes
        # cls_k.shape : b, h, w, 9
        # reg_k : b, h, w, 36
        rpn_rois = rpn_to_roi(cls_k[b,:], reg_k[b,:], no_anchors=num_anchors,  all_possible_anchor_boxes=all_possible_anchor_boxes_tensor.clone() )
        # can't concatenate batch 
        # no of boxes may vary across the batch 
        img_data["boxes"] = boxes[b] // downscale
        img_data['labels'] = labels[b]
        # X2 are qualified anchor boxes from model_rpn (converted anochors)
        # Y1 are the label, Y1[-1] is the background bounding box (negative bounding box), ambigous (neutral boxes are eliminated < min overlap thresold)
        # Y2 
        X2, Y1, Y2, IouS = calc_iou(rpn_rois, img_data, class_mapping=config.label_map )
        break 
        # If X2 is None means there are no matching bboxes
        if X2 is None:
            rpn_accuracy_rpn_monitor.append(0)
            rpn_accuracy_for_epoch.append(0)
            continue

    neg_samples = torch.where(Y1[:, -1] == 1)[0]
    pos_samples = torch.where(Y1[:, -1] == 0)[0]
    
    
    rpn_accuracy_rpn_monitor.append(pos_samples.size(0))
    rpn_accuracy_for_epoch.append(pos_samples.size(0))

    db = Dataset_roi(pos=pos_samples , neg= neg_samples)
    roi_loader = DataLoader(db, shuffle=True,  
        batch_size=args.n_roi // 2, num_workers=args.workers, pin_memory=pin_memory, drop_last=False)

    for i,j in enumerate(roi_loader):
        pos = j[0]
        neg = j[1]
        if pos == []:
            rois = X2[neg]
            base_x[b].unsqueeze(0)

        elif neg == []:





        
        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
        # Y1: one hot code for bboxes from above => x_roi (X)
        # Y2: corresponding labels and corresponding gt bboxes
        



         






         
for epoch_num in range(num_epochs):
    r_epochs += 1
    while True:
        if len(rpn_accuracy_rpn_monitor) == epoch_length :
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
#                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
            if mean_overlapping_bboxes == 0:
                print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

        
        
    
        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]

        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]
        iter_num += 1


        if iter_num == epoch_length:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if C.verbose:
                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                print('Loss RPN regression: {}'.format(loss_rpn_regr))
                print('Loss Detector classifier: {}'.format(loss_class_cls))
                print('Loss Detector regression: {}'.format(loss_class_regr))
                print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                print('Elapsed time: {}'.format(time.time() - start_time))
                elapsed_time = (time.time()-start_time)/60

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            iter_num = 0
            start_time = time.time()

            if curr_loss < best_loss:
                if C.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                best_loss = curr_loss
                model_all.save_weights(C.model_path)

            new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                       'class_acc':round(class_acc, 3), 
                       'loss_rpn_cls':round(loss_rpn_cls, 3), 
                       'loss_rpn_regr':round(loss_rpn_regr, 3), 
                       'loss_class_cls':round(loss_class_cls, 3), 
                       'loss_class_regr':round(loss_class_regr, 3), 
                       'curr_loss':round(curr_loss, 3), 
                       'elapsed_time':round(elapsed_time, 3), 
                       'mAP': 0}

            record_df = record_df.append(new_row, ignore_index=True)
            record_df.to_csv(record_path, index=0)
            break

print('Training complete, exiting.')





# temp = next(iter(dataset_train))
# sanity check 
# list(train_loader)

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

classifier_min_overlap = 0.1
classifier_max_overlap = 0.5
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

epsilon = 1e-4

num_rois = 4 # Number of RoIs to process at once.







model_all.compile(optimizer='sgd', loss='mae')




total_epochs = 100
r_epochs = 100

epoch_length = 1000
num_epochs = 40
iter_num = 0

total_epochs += num_epochs

losses = np.zeros((epoch_length, 5))

if len(record_df)==0:
    best_loss = np.Inf
else:
    best_loss = np.min(r_curr_loss)






# with open('image.pickle', 'wb') as handle:
#     pickle.dump(list(image.numpy()), handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('boxes.pickle', 'wb') as handle:
#     pickle.dump( list(boxes[0].numpy())  , handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('labels.pickle', 'wb') as handle:
#     pickle.dump( list(labels[0].numpy())  , handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('y_rpn_regr.pickle', 'wb') as handle:
#     pickle.dump( list(y_rpn_regr.numpy())  , handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('y_is_box_label.pickle', 'wb') as handle:
#     pickle.dump( list(y_is_box_label.numpy())  , handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('image.pickle', 'rb') as handle:
    image = pickle.load(handle)


with open('boxes.pickle', 'rb') as handle:
    boxes = pickle.load(handle)

with open('labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)

with open('y_is_box_label.pickle', 'rb') as handle:
    y_is_box_label = pickle.load(handle)

with open('y_rpn_regr.pickle', 'rb') as handle:
    y_rpn_regr = pickle.load(handle)


image = torch.tensor(image)
boxes = torch.tensor(boxes)
labels = torch.tensor(labels)
y_is_box_label = torch.tensor(y_is_box_label)
y_rpn_regr = torch.tensor(y_rpn_regr)

y_is_box_label = y_is_box_label.squeeze(0)
y_rpn_regr = y_rpn_regr.squeeze(0)
num_pos = 31



