
import os
import argparse
import math 

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from  model import Model_RPN , Classifier
from tools import * 
from utils import WarmupMultiStepLR  , save_checkpoint , load_checkpoint

from dataset import Dataset , collate_fn , Dataset_roi
import torchvision.transforms as transforms

from torch.autograd import Variable
from loss import rpn_loss_regr , rpn_loss_cls_fixed_num  , class_loss_cls , class_loss_regr

from plot import save_evaluations_image


class Config(object):
    """docstring for config"""
    def __init__(self):
        super(Config, self).__init__()
        self.voc_labels = ('schizont', 'gametocyte', 'trophozoite', 'red blood cell', 'difficult', 'ring', 'leukocyte')
        self.voc_labels += ('bg',)
        self.label_map = {k: v for v, k in enumerate(self.voc_labels)}
        self.rev_label_map = {v: k for k, v in self.label_map.items()}  # Inverse mapping
        # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        self.distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

config = Config()

parser = argparse.ArgumentParser(description='Faster RCNN (Custom Dataset)')
# basic running parameters 
parser.add_argument('--train-batch', default=2, type=int,
                    help="train batch size")
parser.add_argument('--workers', default=8, type=int,
                    help="# of workers, keep it greater than 4")
parser.add_argument('--seed', default=1, type=int,
                    help="# seed")
parser.add_argument('--gpu-devices', default="0,1", type=str,
                    help="# of gpu devices")
parser.add_argument('--display-rpn', default=1, type=int,
                    help="display frequency of performance of rpn model")
parser.add_argument('--display-class', default=20, type=int,
                    help="display frequency of performance of classification model")
parser.add_argument('--save-evaluations', action='store_true', default=False,
                    help="if used, will save validation set images with boxes")
parser.add_argument('--pretrained', action='store_true', default=False,
                    help="if used, load pretrained weights")

# training related specs
parser.add_argument('-m-epochs', '--max-epochs', type=int, default=20,
                    help="maximum number of epochs")
parser.add_argument('-d', '--dataset', type=str, default='./',
                    help="path of the datatset...")
parser.add_argument('-lr-rpn', '--learning-rate-rpn', default=0.0035, type=float,
                    help="initial learning rate for model-rpn")
parser.add_argument('-lr-classifier', '--learning-rate-classifier', default=0.0035, type=float,
                    help="initial learning rate for model-rpn")
parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="weight decay for the model")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="gamma for learning rate schedule")

# image specs
parser.add_argument('--height', type=int, default=800,
                    help="height of an image (default: 800)")
parser.add_argument('--width', type=int, default=600,
                    help="width of an image (default: 600)")
parser.add_argument('--data-format', type=str, default='bg_last',
                    help="code is written with assumption that backgound class is at last, 'bg_first' handles the other way round")

parser.add_argument('--anchor-sizes', default=None, type=list , help="anchor box sizes ")
parser.add_argument('--anchor-ratio', default=[1,0.5,2], type=list , help="anchor box ratios " )
# anchor box specs
parser.add_argument('--rpn-min-overlap', default=0.3, type=float,
                    help="min overlap")
parser.add_argument('--rpn-max-overlap', default=0.7, type=float,
                    help="max overlap with ground truth ")
parser.add_argument('--classifier-min-overlap', default=0.1, type=float,
                    help="min overlap for anchor box qualification")
parser.add_argument('--classifier-max-overlap', default=0.5, type=float,
                    help="frequency thresold after which anchor box is declared positive")
parser.add_argument('--thresold-num-region', default=300, type=int,
                    help="limiting the number of positive + negative anchor boxes to thresold-num-region")
parser.add_argument('--classifier-regr-std', default=[8.0, 8.0, 4.0, 4.0], type=list , help="scaling factor for tx,ty,tw and th for model classifier" )
parser.add_argument('--std_scaling', default=4.0, type=float,
                    help="scalling factor for regression ")
parser.add_argument('--n-roi', type=int, default=20,
                    help="number of roi to train classifiers with")

# loss scaling factor
parser.add_argument('--lambda-rpn-regr', default=1.0, type=float,
                    help="scaling factor for the model rpn regression")
parser.add_argument('--lambda-rpn-class', default=1.0, type=float,
                    help="scaling factor for the model rpn classification loss")
parser.add_argument('--lambda-cls-regr', default=1.0, type=float,
                    help="scaling factor for the model classifier regression loss")
parser.add_argument('--lambda-cls-class', default=1.0, type=float,
                    help="scaling factor for the model classifier classification loss")

# directory
parser.add_argument('-s', '--save_dir', type=str, default='models/',
                    help="path of the model weights")



args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(1)

# args.gpu_devices = "0,1"
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()
pin_memory = True if use_gpu else False
cudnn.benchmark = True

if use_gpu:
    device = 'cuda'
else:
    device = 'cpu'



height = args.height
width = args.width


if args.save_evaluations:
    denormalize = {}

    std = torch.tensor([[0.229, 0.224, 0.225]]).expand(height,3).unsqueeze(0).expand(width,height,3)
    std = std.permute(2,1,0)

    mean = torch.tensor([ [0.485, 0.456, 0.406] ]).expand(height,3).unsqueeze(0).expand(width,height,3)
    mean = mean.permute(2,1,0)

    denormalize['mean'] = mean.to(device=device)
    denormalize['std'] = std.to(device=device)


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
rpm = RPM(anchor_sizes , anchor_ratios, valid_anchors, config.rev_label_map, rpn_max_overlap=args.rpn_max_overlap , rpn_min_overlap= args.rpn_min_overlap , num_regions = args.thresold_num_region )

dataset_train =  Dataset(data_folder=args.dataset, rpm=rpm, split='TRAIN', std_scaling=args.std_scaling, image_resize_size= (height, width),  debug= False , data_format= args.data_format)
dataset_test =  Dataset(data_folder=args.dataset, rpm=rpm, split='TEST', std_scaling=args.std_scaling, image_resize_size= (height, width),  debug= False , data_format= args.data_format )

# keep the number of workers greater than 4
train_loader = DataLoader(
    dataset_train, shuffle=True,  collate_fn=collate_fn, 
    batch_size=args.train_batch, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)


test_loader = DataLoader(
    dataset_test, shuffle=True,  collate_fn=collate_fn, 
    batch_size=1, num_workers=args.workers, pin_memory=pin_memory, drop_last=True)



# temp = next(iter(dataset_train))
# sanity check 
# list(train_loader)

state = None 
if args.pretrained:
    state=  load_checkpoint(save_dir=args.save_dir , device=device)
    if state == None :
        print("==== No Pretrained weights found")


if state == None :
    print("loading from scratch \n\n\n\n") 
    best_error = math.inf
    start_epoch  = -1 

    model_rpn = Model_RPN(num_anchors= len(anchor_sizes) * len(anchor_ratios) )
    model_classifier = Classifier(num_classes=  len(config.voc_labels) )


    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model_rpn = nn.DataParallel(model_rpn , device_ids= [1,2,3])
      # DO NOT PARALLIZE model_classifier on multiple GPUs.... 
      # pytorch bug screws up the output
      model_classifier = nn.DataParallel(model_classifier , device_ids= [0])

    # model_rpn = model_rpn.to(device)
    model_rpn.to(f'cuda:{model_rpn.device_ids[0]}')
    model_classifier.to(f'cuda:{model_classifier.device_ids[0]}')
    
    # model_rpn = model_rpn.to(device)
    # model_classifier = model_classifier.to(device)



    weight_decay = args.weight_decay 
    params_class = []
    for key, value in model_classifier.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.learning_rate_classifier
        params_class += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    params_rpn = []
    for key, value in model_rpn.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.learning_rate_rpn
        params_rpn += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    optimizer_model_rpn = torch.optim.Adam(params_rpn)
    optimizer_classifier = torch.optim.Adam(params_class)

else:
    print("loading pretrained weights \n\n\n\n")
    best_error = state['best_error']

    model_rpn = state['model_rpn'].to(device=device)
    model_classifier = state['model_classifier'].to(device=device)

    optimizer_model_rpn = state['optimizer_model_rpn']
    optimizer_classifier = state['optimizer_classifier']
    
    start_epoch  = state['epoch']

    # if pretrained optimizer was from CPU 
    # https://github.com/pytorch/pytorch/issues/2830
    if use_gpu:
        for state in optimizer_model_rpn.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in optimizer_classifier.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()


model_rpn_cuda = f'cuda:{model_rpn.device_ids[0]}'
model_classifier_cuda = f'cuda:{model_classifier.device_ids[0]}'


scheduler_rpn = WarmupMultiStepLR(optimizer_model_rpn, milestones=[40, 70], gamma=args.gamma, warmup_factor=0.01, warmup_iters=10)
scheduler_class = WarmupMultiStepLR(optimizer_classifier, milestones=[40, 70], gamma=args.gamma, warmup_factor=0.01, warmup_iters=10)

all_possible_anchor_boxes = default_anchors(out_h=out_h, out_w=out_w, anchor_sizes=anchor_sizes , anchor_ratios=anchor_ratios , downscale=16)
all_possible_anchor_boxes_tensor = torch.tensor(all_possible_anchor_boxes).to(device=device)


def train(epoch):
    print("\n\nTraining epoch {}\n\n".format(epoch))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    
    regr_rpn_loss= 0 
    class_rpn_loss =0 
    total_rpn_loss = 0 

    regr_class_loss= 0
    class_class_loss =0  
    total_class_loss = 0     

    count_rpn  = 0 
    count_class = 0 


    for i,(image, boxes, labels , temp, num_pos) in enumerate(train_loader):
        count_rpn +=1

        y_is_box_label = temp[0].to(device=model_rpn_cuda)
        y_rpn_regr = temp[1].to(device=model_rpn_cuda)
        image = Variable(image).to(device=model_rpn_cuda)
        boxes = boxes
        base_x , cls_k , reg_k = model_rpn(image)
        
        l1 = rpn_loss_regr(y_true=y_rpn_regr, y_pred=reg_k , y_is_box_label=y_is_box_label , lambda_rpn_regr=args.lambda_rpn_regr , device=model_rpn_cuda)
        l2 = rpn_loss_cls_fixed_num(y_pred = cls_k , y_is_box_label= y_is_box_label , lambda_rpn_class = args.lambda_rpn_class)
        
        regr_rpn_loss += l1.item() 
        class_rpn_loss += l2.item() 
        loss = l1 + l2 
        total_rpn_loss += loss.item()
        
        optimizer_model_rpn.zero_grad()
        loss.backward()
        optimizer_model_rpn.step()                        

        with torch.no_grad():

            base_x , cls_k , reg_k = model_rpn(image)
            
            cls_k = cls_k.to(device=model_classifier_cuda)
            reg_k = reg_k.to(device=model_classifier_cuda)
            base_x = base_x.to(device=model_classifier_cuda)
            

        for b in range(args.train_batch):
            img_data = {}
            with torch.no_grad():
                # Convert rpn layer to roi bboxes
                # cls_k.shape : b, h, w, 9
                # reg_k : b, h, w, 36
                # model_classifier_cuda
                # cls_k[b,:].shape == [13, 10, 9]
                # reg_k[b,:].shape == [13, 10, 36]
                # num_anchors = 9
                # all_possible_anchor_boxes_tensor.shape == [4, 13, 10, 9]
                
                rpn_rois = rpn_to_roi(cls_k[b,:].cpu(), reg_k[b,:].cpu(), no_anchors=num_anchors,  all_possible_anchor_boxes=all_possible_anchor_boxes_tensor.cpu().clone() )
                rpn_rois.to(device=model_classifier_cuda)
                # can't concatenate batch 
                # no of boxes may vary across the batch 
                img_data["boxes"] = boxes[b].to(device=model_classifier_cuda) // downscale
                img_data['labels'] = labels[b]

                # rpn_rois : 300, 4
                # img_data["boxes"].shape : 68,4
                # len(img_data["labels"]) : 68

                # X2 are qualified anchor boxes from model_rpn (converted anochors)
                # Y1 are the label, Y1[-1] is the background bounding box (negative bounding box), ambigous (neutral boxes are eliminated < min overlap thresold)
                # Y2 is concat of 1 , tx, ty, tw, th and 0, tx, ty, tw, th 
                X2, Y1, Y2, _ = calc_iou(rpn_rois, img_data, class_mapping=config.label_map )
                
                X2 = X2.to(device=model_classifier_cuda)
                Y1 = Y1.to(device=model_classifier_cuda)
                Y2 = Y2.to(device=model_classifier_cuda)
            
                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = torch.where(Y1[:, -1] == 1)[0]
                pos_samples = torch.where(Y1[:, -1] == 0)[0]
                rpn_accuracy_rpn_monitor.append(pos_samples.size(0))
                rpn_accuracy_for_epoch.append(pos_samples.size(0))
            
            db = Dataset_roi(pos=pos_samples.cpu() , neg= neg_samples.cpu())
            roi_loader = DataLoader(db, shuffle=True,  
                batch_size=args.n_roi // 4, num_workers=args.workers//2, pin_memory=False, drop_last=False)
            # list(roi_loader)
            for j,potential_roi in enumerate(roi_loader):
                pos = potential_roi[0]
                neg = potential_roi[1]
                if type(pos) == list :
                    rois = X2[neg]
                    rpn_base = base_x[b].unsqueeze(0)
                    Y11 = Y1[neg]
                    Y22 = Y2[neg]
                    # out_class : args.n_roi // 2 , # no of class
                elif type(neg) == list :
                    rois = X2[pos]
                    rpn_base = base_x[b].unsqueeze(0)
                    #out_class :  args.n_roi // 2 , # no of class
                    Y11 = Y1[pos]
                    Y22 = Y2[pos]
                else:
                    ind = torch.cat([pos,neg])
                    rois = X2[ind]
                    rpn_base = base_x[b].unsqueeze(0)
                    #out_class:  args.n_roi , # no of class
                    Y11 = Y1[ind]
                    Y22 = Y2[ind]
                
                # IF YOU ARE NOT SEEING THESE SHAPES THEN SOMETHING IS WRONG
                # Y11.shape  = 20,8
                # Y22.shape = 20,56
                # out_class.shape = 20,8
                # out_regr.shape = 20,56
                # rois.shape = 20, 4 
                # rpn_base.shape  = torch.Size([1, 2048, 50, 38])
                count_class += 1
                rois = Variable(rois).to(device=model_classifier_cuda)
                out_class , out_regr = model_classifier(base_x = rpn_base , rois= rois )
                # torch.Size([5, 2048, 7, 7]) torch.Size([5, 4])

                l3 = class_loss_cls(y_true=Y11, y_pred=out_class , lambda_cls_class=args.lambda_cls_class)
                l4 = class_loss_regr(y_true=Y22, y_pred= out_regr , lambda_cls_regr= args.lambda_cls_regr)

                regr_class_loss += l4.item()
                class_class_loss += l3.item()   

                loss = l3 + l4 
                total_class_loss += loss.item()
                
                
                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_classifier.step()

                if count_class % args.display_class == 0 :
                    if count_class == 0 :
                        print('[Classifier] RPN Ex: {}-th ,Batch : {}, Anchor Box: {}-th, Classifier Model Classification loss: {} Regression loss: {} Total Loss: {}'.format(i,b,j,0,0,0))
                    else:
                        print('[Classifier] RPN Ex: {}-th ,Batch : {}, Anchor Box: {}-th, Classifier Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(i,b,j, class_class_loss / count_class, regr_class_loss / count_class ,total_class_loss/ count_class ))

        if i % args.display_rpn == 0 :
            if len(rpn_accuracy_rpn_monitor) == 0 :
                print('[RPN] RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            else:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                print('[RPN] Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes)) 
            print('[RPN] RPN Ex: {}-th RPN Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(i ,class_rpn_loss / count_rpn, regr_rpn_loss / count_rpn ,total_rpn_loss/ count_rpn ))

    print("-- END OF EPOCH -- {}".format(epoch)) 
    print("------------------------------" ) 
    print('[RPN] RPN Ex: {}-th RPN Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(i ,class_rpn_loss / count_rpn, regr_rpn_loss / count_rpn ,total_rpn_loss/ count_rpn ))
    if count_class == 0 :
        print('[Classifier] RPN Ex: {}-th ,Batch : {}, Anchor Box: {}-th, Classifier Model Classification loss: {} Regression loss: {} Total Loss: {}'.format(i,b,j,0,0,0))
    else:
        print('[Classifier] RPN Ex: {}-th ,Batch : {}, Anchor Box: {}-th, Classifier Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(i,b,j, class_class_loss / count_class, regr_class_loss / count_class ,total_class_loss/ count_class ))
    if len(rpn_accuracy_rpn_monitor) == 0 :
        print('[RPN] RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
    else:
        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
        print('[RPN] Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes)) 
    print('Total Loss  {}'.format(  total_class_loss/ count_class + total_rpn_loss/ count_rpn ))
    print("------------------------------" ) 



def test(epoch):
    print("================================")
    print("Testing after epoch {}".format(epoch))
    print("================================")
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    
    regr_rpn_loss= 0 
    class_rpn_loss =0 
    total_rpn_loss = 0 

    regr_class_loss= 0
    class_class_loss =0  
    total_class_loss = 0     

    count_rpn  = 0 
    count_class = 0 

    total_count = 0

    for i,(image, boxes, labels , temp, num_pos) in enumerate(test_loader):

        count_rpn +=1
        
        y_is_box_label = temp[0].to(device=model_rpn_cuda)
        y_rpn_regr = temp[1].to(device=model_rpn_cuda)
        image = Variable(image).to(device=model_rpn_cuda)

        base_x , cls_k , reg_k = model_rpn(image)
        l1 = rpn_loss_regr(y_true=y_rpn_regr, y_pred=reg_k , y_is_box_label=y_is_box_label , lambda_rpn_regr=args.lambda_rpn_regr, device=model_rpn_cuda)
        l2 = rpn_loss_cls_fixed_num(y_pred = cls_k , y_is_box_label= y_is_box_label , lambda_rpn_class = args.lambda_rpn_class)
        
        regr_rpn_loss += l1.item() 
        class_rpn_loss += l2.item() 
        loss = l1 + l2 
        total_rpn_loss += loss.item()
        
        base_x , cls_k , reg_k = model_rpn(image)

        for b in range(image.size(0)):
            img_data = {}
            rpn_rois = rpn_to_roi(cls_k[b,:].cpu(), reg_k[b,:].cpu(), no_anchors=num_anchors,  all_possible_anchor_boxes=all_possible_anchor_boxes_tensor.cpu().clone() )
            rpn_rois.to(device=model_classifier_cuda)

            img_data["boxes"] = boxes[b].to(device=model_classifier_cuda) // downscale
            img_data['labels'] = labels[b]

            X2, Y1, Y2, _ = calc_iou(rpn_rois, img_data, class_mapping=config.label_map )
            
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            X2 = X2.to(device=model_classifier_cuda)
            Y1 = Y1.to(device=model_classifier_cuda)
            Y2 = Y2.to(device=model_classifier_cuda)

            count_class += 1 
            rpn_base = base_x[b].unsqueeze(0)
            out_class , out_regr = model_classifier(base_x = rpn_base , rois= X2 )
            
            l3 = class_loss_cls(y_true=Y1, y_pred=out_class , lambda_cls_class=args.lambda_cls_class)
            l4 = class_loss_regr(y_true=Y2, y_pred= out_regr , lambda_cls_regr= args.lambda_cls_regr)
            loss = l3 + l4 
            class_class_loss += l3.item()   
            regr_class_loss += l4.item()
            total_class_loss += loss.item()

            if args.save_evaluations :
                total_count += 1
                if total_count % 100 == 0 :
                    predicted_boxes = X2
                    predicted_boxes[:,2]  = predicted_boxes[:,2]  + predicted_boxes[:,0]
                    predicted_boxes[:,3]  = predicted_boxes[:,3]  + predicted_boxes[:,1]
                    predicted_boxes = predicted_boxes * downscale
                    
                    temp_img = (denormalize['std'].cpu() * image[b].cpu()) + denormalize['mean'].cpu()  
                    save_evaluations_image(image=temp_img, boxes=predicted_boxes, labels=Y1, count=total_count, config=config , save_dir=args.save_dir)
    
    if count_class == 0 :
        count_class = 1
        total_class_loss = 0 
        print('[Test Accuracy] Classifier Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(0, 0 ,0 ))
    else:
        print('[Test Accuracy] Classifier Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(class_class_loss / count_class, regr_class_loss / count_class ,total_class_loss/ count_class ))

    print('[Test Accuracy] RPN Model Classification loss: {} Regression loss: {} Total Loss: {} '.format(class_rpn_loss / count_rpn, regr_rpn_loss / count_rpn ,total_rpn_loss/ count_rpn ))
    if len(rpn_accuracy_rpn_monitor) == 0 :
        print('[Test Accuracy] RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')        
    else:
        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
        print('[Test Accuracy] Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes)) 
    
    return total_class_loss/ count_class + total_rpn_loss/ count_rpn


model_rpn.eval()
model_classifier.eval()  
total_loss = test(0)
print(os.environ['CUDA_VISIBLE_DEVICES'])
for i in range(start_epoch + 1 , args.max_epochs):
    model_rpn.train()
    model_classifier.train()    
    train(i)
    scheduler_rpn.step()
    scheduler_class.step()
    model_rpn.eval()
    model_classifier.eval()  
    total_loss = test(i)
    if total_loss < best_error : 
        save_checkpoint(i, model_rpn, model_classifier, optimizer_model_rpn, optimizer_classifier , best_error, save_dir=args.save_dir)

    print("=== {} === ".format(total_loss))  
    
print('Training complete, exiting.')

# export CUDA_VISIBLE_DEVICES=3  
# python main.py --train-batch=2 --workers=1 \
# --display-rpn=100 --display-class=100 --save-evaluations --max-epochs=100 \
# --height=200 --width=150 --save_dir="/nfs/bigcornea/add_disk0/pathak/biodata2/BBBC041/"



