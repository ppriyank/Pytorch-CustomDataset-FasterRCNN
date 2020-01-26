import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable




class Model(nn.Module):
    def __init__(self, num_classes, anchor_box_scales=[512,256,128], anchor_ratio=[1,0.5,2], **kwargs):
        super(Model, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride = (1,1)
        resnet50.layer4[0].downsample[0].stride = (1,1)
        
        self.num_anchors = len(anchor_ratio) * len(anchor_box_scales)
        self.feat_dim = 2048
        self.middle_dim = self.feat_dim  // 4
        self.pooling_regions = 7
        self.num_rois=4 

        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        self.rpn_c1 = nn.Conv2d(self.feat_dim, self.middle_dim, 3 , padding=1 )
        self.relu1 = nn.ReLU()

        self.rpn_cls = nn.Conv2d(self.middle_dim, self.num_anchors, 1  )
        self.rpn_reg = nn.Conv2d(self.middle_dim, 4* self.num_anchors, 1 )
        
        self.sigmoid1  = nn.Sigmoid()        

    def forward(self, x):
        b = x.size(0)
        h = x.size(2)
        w = x.size(3)
        c = x.size(1)

        # torch.Size([1, 2048, 19, 25])
        base_x = self.base(x)
        
        #### RPN 
        # torch.Size([1, 512, 19, 25])
        x = self.relu1(self.rpn_c1(base_x))
        
        # torch.Size([1, 9, 19, 25])
        cls_k = self.sigmoid1(self.rpn_cls(x))
        
        # torch.Size([1, 36, 19, 25])
        reg_k = self.rpn_reg(x)

        # [cls_k , reg_k, base_x]
        
        # x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(b,t,-1)
        # x=x.permute(0,2,1)
        # f = F.avg_pool1d(x,t)
        # f = f.view(b, self.feat_dim)
        # if not self.training:
        #     return f
        # y = self.classifier(f)


# import torch 
# import torchvision
# from torch import nn

# x = torch.rand([1,3,300,400])
# resnet50 = torchvision.models.resnet50(pretrained=True)
# resnet50.layer4[0].conv2.stride = (1,1)
# resnet50.layer4[0].downsample[0].stride = (1,1)

# num_anchors = 9
# feat_dim = 2048
# middle_dim = feat_dim  // 4

# base = nn.Sequential(*list(resnet50.children())[:-2])

# rpn_c1 = nn.Conv2d(feat_dim, middle_dim, 3 , padding=1 )
# relu1 = nn.ReLU()

# rpn_cls = nn.Conv2d(middle_dim, num_anchors, 1  )
# rpn_reg = nn.Conv2d(middle_dim, 4* num_anchors, 1 )

# sigmoid1  = nn.Sigmoid()        



# b = x.size(0)
# h = x.size(2)
# w = x.size(3)

# base_x = base(x)
# #### RPN 
# #torch.Size([32, 2048, 50, 38])
# x = relu1(rpn_c1(base_x))
# #torch.Size([32, 512, 50, 38])
# # x = torch.rand (1,512,50,38)

# cls_k = sigmoid1(rpn_cls(x))
# #torch.Size([b, k, 50, 38])

# reg_k = rpn_reg(x)
     

import torch.nn as nn
# rois = torch.shape : (4,4)
outputs = []
rois  = torch.tensor([[12, 20, 2, 2] , [4, 4, 4, 4] , [5,2, 6 , 7] , [8,12, 12, 20]])
base_x = torch.rand([1,2048,19,25])
feat_dim = 2048
pooling_regions = 7
nb_classes = 6 

red_conv_roi = nn.Conv2d(feat_dim, feat_dim//4 , 1 )

d1 = nn.Linear(feat_dim//4 * pooling_regions * pooling_regions , feat_dim * 2 , bias=True)
relu_d1 = nn.ReLU()
drop_d1 = nn.Dropout(p=0.5, inplace=False)

d2 = nn.Linear(feat_dim * 2 , feat_dim , bias=True)
relu_d2 = nn.ReLU()
drop_d2 = nn.Dropout(p=0.5, inplace=False)

d3 = nn.Linear(feat_dim  , nb_classes , bias=False)
softmax_d3 = nn.Softmax(1)

d4 = nn.Linear(feat_dim  , 4 * (nb_classes-1) , bias=False)



for rid in range(rois.size(0)) :
    x , y, h , w = rois[rid]
    x , y, h , w = x.int() , y.int(), h.int() , w.int() 
    cropped_image = base_x[:,:, x:x+w, y:y+h]
    resized_image = (F.adaptive_avg_pool2d(Variable(cropped_image,volatile=True), ( pooling_regions ,pooling_regions ) ))
    outputs.append(resized_image.unsqueeze(0))

    
out_roi_pool = torch.cat(outputs,1)
out_roi_pool = out_roi_pool.view(4, feat_dim, 7, 7 )

red_out_roi_pool=  red_conv_roi(out_roi_pool)
red_out_roi_pool = red_out_roi_pool.view(4,-1)

red_out_roi_pool = drop_d1(relu_d1(d1(red_out_roi_pool)))
red_out_roi_pool = drop_d2(relu_d2(d2(red_out_roi_pool)))

out_class = softmax_d3(d3(red_out_roi_pool))
out_regr = d4(red_out_roi_pool)







out_roi_pool.
