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
        

def classifier_layer(base_layers, input_rois, , nb_classes = 4):
    """Create a classifier layer
    
    Args:
        base_layers: vgg
        input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        num_rois: number of rois to be processed in one time (4 in here)

    Returns:
        list(out_class, out_regr)
        out_class: classifier layer output
        out_regr: regression layer output
    """

    input_shape = (num_rois,7,7,512)

    

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]






class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):
        
    def call(self, x, mask=None):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)
                

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))