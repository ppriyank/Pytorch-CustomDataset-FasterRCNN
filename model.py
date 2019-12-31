import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable




class Model(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Model, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride = (1,1)
        resnet50.layer4[0].downsample[0].stride = (1,1)
        
        self.num_anchors = 9        
        self.feat_dim = 2048
        self.middle_dim = self.feat_dim  // 4

        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        self.rpn_c1 = nn.Conv2d(self.feat_dim, self.middle_dim, 3 , padding=1 )
        self.relu1 = nn.ReLU()

        self.rpn_cls = nn.Conv2d(self.middle_dim, self.num_anchors, 1 , padding=1 )
        self.rpn_reg = nn.Conv2d(self.middle_dim, 4* self.num_anchors, 1 , padding=1 )
        
        self.sigmoid1  = nn.Sigmoid()        

    def forward(self, x):
        b = x.size(0)
        h = x.size(2)
        w = x.size(3)

        x = self.base(x)
        #torch.Size([32, 2048, 50, 38])
        x = self.relu1(self.rpn_c1(x))
        #torch.Size([32, 512, 50, 38])
        # x = torch.rand (1,512,50,38)

        cls_k = self.sigmoid1(self.rpn_cls(x))
        reg_k = self.rpn_cls(x)

        
        rpn_c1 = nn.Conv2d(self.feat_dim, self.middle_dim, 3 )        

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
