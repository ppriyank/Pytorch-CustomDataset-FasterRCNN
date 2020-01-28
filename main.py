
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


from  model import Model_RPN
from tools import * 

from dataset import Dataset , collate_fn
import torchvision.transforms as transforms

from torch.autograd import Variable
from loss import rpn_loss_regr , rpn_loss_cls_fixed_num 


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
parser.add_argument('--train-batch', default=2, type=int,
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


train_loader = DataLoader(
    dataset_train, shuffle=True,  collate_fn=collate_fn, 
    batch_size=args.train_batch, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)


test_loader = DataLoader(
    dataset_test, shuffle=True,  collate_fn=collate_fn, 
    batch_size=1, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)




model_rpn = Model_RPN(num_anchors= len(anchor_sizes) * len(anchor_ratios) )

all_possible_anchor_boxes = default_anchors(out_h=50, out_w=38, anchor_sizes=anchor_sizes , anchor_ratios=anchor_ratios , downscale=16)
all_possible_anchor_boxes_tensor = torch.tensor(all_possible_anchor_boxes)
# all_possible_anchor_boxes_tensor = all_possible_anchor_boxes_tensor.unsqueeze(0).repeat(args.train_batch, 1,1,1,1)
# torch.Size([4, 50, 38, 9])


for i,(image, boxes, labels , temp, num_pos) in enumerate(train_loader):
            break

y_is_box_label = temp[0]
y_rpn_regr = temp[1]


image = Variable(image)


weight_decay = 0.0005
base_learning_rate =  0.0035
params_rpn = []
for key, value in model_rpn.named_parameters():
    if not value.requires_grad:
        continue
    lr = base_learning_rate
    params_rpn += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


optimizer_model_rpn = torch.optim.Adam(params_rpn)
base_x , cls_k , reg_k = model_rpn(image)


l1 = rpn_loss_regr(y_true=y_rpn_regr, y_pred=reg_k , y_is_box_label=y_is_box_label)
l2 = rpn_loss_cls_fixed_num(y_pred = cls_k , y_is_box_label= y_is_box_label)

loss = l1 + l2 
optimizer_model_rpn.zero_grad()
loss.backward()
optimizer_model_rpn.step()

with torch.no_grad():
    base_x , cls_k , reg_k = model_rpn(image)
    for b in range(args.train_batch):
        temp = rpn_to_roi(cls_k[b,:], reg_k[b,:], no_anchors=num_anchors,  all_possible_anchor_boxes=all_possible_anchor_boxes_tensor.clone() )
        # can't concatenate batch 
        # no of boxes may vary across the batch 

        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
        # Y1: one hot code for bboxes from above => x_roi (X)
        # Y2: corresponding labels and corresponding gt bboxes
        X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)



         



# Convert rpn layer to roi bboxes
# cls_k.shape : b, h, w, 9
# reg_k : b, h, w, 36



         
for epoch_num in range(num_epochs):
    r_epochs += 1
    while True:
        if len(rpn_accuracy_rpn_monitor) == epoch_length :
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
#                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
            if mean_overlapping_bboxes == 0:
                print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

        for i,(image, boxes, labels , temp, num_pos) in enumerate(train_loader):
            break

        y_is_box_label = temp[0]
        y_rpn_regr = temp[1]

        
        
        
        # If X2 is None means there are no matching bboxes
        if X2 is None:
            rpn_accuracy_rpn_monitor.append(0)
            rpn_accuracy_for_epoch.append(0)
            continue
        
        # Find out the positive anchors and negative anchors
        neg_samples = np.where(Y1[0, :, -1] == 1)
        pos_samples = np.where(Y1[0, :, -1] == 0)

        if len(neg_samples) > 0:
            neg_samples = neg_samples[0]
        else:
            neg_samples = []

        if len(pos_samples) > 0:
            pos_samples = pos_samples[0]
        else:
            pos_samples = []

        rpn_accuracy_rpn_monitor.append(len(pos_samples))
        rpn_accuracy_for_epoch.append((len(pos_samples)))

        if C.num_rois > 1:
            # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
            if len(pos_samples) < C.num_rois//2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
            
            # Randomly choose (num_rois - num_pos) neg samples
            try:
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
            
            # Save all the pos and neg samples in sel_samples
            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            # in the extreme case where num_rois = 1, we pick a random pos or neg sample
            selected_pos_samples = pos_samples.tolist()
            selected_neg_samples = neg_samples.tolist()
            if np.random.randint(0, 2):
                sel_samples = random.choice(neg_samples)
            else:
                sel_samples = random.choice(pos_samples)

        # training_data: [X, X2[:, sel_samples, :]]
        # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
        #  X                     => img_data resized image
        #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
        #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
        #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
        loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]

        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]

        iter_num += 1

        progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                  ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

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

epsilon = 1e-4

num_rois = 4 # Number of RoIs to process at once.





optimizer_classifier = torch.optim.Adam(classifier_params)

optimizer_classifier = torch.optim.Adam(classifier_params)


model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')




total_epochs = 100
r_epochs = 100

epoch_length = 1000
num_epochs = 40
iter_num = 0

total_epochs += num_epochs

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

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



