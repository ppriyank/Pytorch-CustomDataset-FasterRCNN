import torch 
import torch.nn as nn 

from utils import tile 

def rpn_loss_regr(y_true, y_pred , y_is_box_label , lambda_rpn_regr = 1.0):
        # Smooth L1 loss function 
        #                    0.5*x*x (if x_abs < 1)
        #                    x_abx - 0.5 (otherwise)

        # y_true [: , : , : , 36]: 4 values per 9 anchor boxes 
        # y_is_box_label [: , : , : , 9] tilled on the last dimension 4 times. label 1 propogated through all 4 values ,
         # similarly other labels replicated 4 times 

        epsilon = 1e-6
        x = y_true - y_pred
        x_abs = torch.abs(x)
        index_small = torch.where(x_abs <= 1 )
        
        x_abs[index_small] = torch.pow(x_abs[index_small], 2) /  2 + 0.5
        x_abs = x_abs - 0.5
        # label 1 means positive box, label 0 means neutral and -1 means negative box
        # clamp min= 0, removes negative box error
        y_is_box_label = tile(y_is_box_label, -1 , 4 ).clamp(min=0)

        loss = (y_is_box_label * x_abs).sum() / (epsilon + y_is_box_label.sum() )
        return lambda_rpn_regr * loss 
        


        
def rpn_loss_cls_fixed_num(y_pred, y_is_box_label , lambda_rpn_class=1.0):
        # torch.abs(y_is_box_label) will keep the positive and negative labels and ignore the 0 labels 
        # y_pred [: ,: ,: , 9 ]
        # y_is_box_label.clamp(min=0) converts the problem into binary cross entropy : postive label =1, negative & neutral label = 0 


        ce=  nn.BCELoss(reduction='none')
        temp = torch.abs(y_is_box_label)
        epsilon = 1e-6 
        ce_loss = ce(y_pred , y_is_box_label.clamp(min=0))
        loss = ( temp * ce_loss ).sum()  / ( epsilon + temp.sum())
        return lambda_rpn_class * loss


# class cross entropy

def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num



def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))



