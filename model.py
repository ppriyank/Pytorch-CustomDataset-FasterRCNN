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

        x = self.base(x)

        #### RPN 
        #torch.Size([32, 2048, 50, 38])
        x = self.relu1(self.rpn_c1(x))
        #torch.Size([32, 512, 50, 38])
        # x = torch.rand (1,512,50,38)

        cls_k = self.sigmoid1(self.rpn_cls(x))
        #torch.Size([b, k, 50, 38])
        reg_k = self.rpn_reg(x)
        #torch.Size([b, 4 *k, 50, 38])

        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)



def iou(a, b):
    # (xmin,ymin,xmax,ymax)
    # invlaid boxes
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    #intersection 
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 - x1 <= 0  or y2 - y1 <=0 :
        intersection = 0 
    else:
        intersection = (x2 - x1) * (y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union =  area_a + area_b - intersection

    if union <= 0 : 
        return 0.0
    else:
        return float(intersection) / float(union + 1e-6)