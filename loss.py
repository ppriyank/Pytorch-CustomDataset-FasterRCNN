import torch 
import torch.nn as nn 

from utils import tile 

def rpn_loss_regr(y_true, y_pred , y_is_box_label , lambda_rpn_regr = 1.0 , epsilon = 1e-6, device='cpu'):
        # Smooth L1 loss function 
        #                    0.5*x*x (if x_abs < 1)
        #                    x_abx - 0.5 (otherwise)

        # y_true [: , : , : , 36]: 4 values per 9 anchor boxes 
        # y_is_box_label [: , : , : , 9] tilled on the last dimension 4 times. label 1 propogated through all 4 values ,
         # similarly other labels replicated 4 times 
        b = y_true.size(0)
        x_abs = y_true - y_pred
        x_abs = torch.abs(x_abs)
        index_small = torch.where(x_abs <= 1 )
        
        x_abs[index_small] = torch.pow(x_abs[index_small], 2) /  2 + 0.5
        x_abs = x_abs - 0.5
        # label 1 means positive box, label 0 means neutral and -1 means negative box
        # clamp min= 0, removes negative box error
        y_is_box_label = tile(y_is_box_label, -1 , 4 , device=device).clamp(min=0)

        loss = (y_is_box_label * x_abs).view(b,-1).sum(1) / (epsilon + y_is_box_label.view(b,-1).sum(1) )
        # loss = (y_is_box_label * x_abs).sum() / (epsilon + y_is_box_label.sum() )
        return lambda_rpn_regr * loss.mean() 
        


        
def rpn_loss_cls_fixed_num(y_pred, y_is_box_label , lambda_rpn_class=1.0 , epsilon = 1e-6 ):
        # torch.abs(y_is_box_label) will keep the positive and negative labels and ignore the 0 labels 
        # y_pred [: ,: ,: , 9 ]
        # y_is_box_label.clamp(min=0) converts the problem into binary cross entropy : postive label =1, negative & neutral label = 0 

        b = y_pred.size(0)
        ce=  nn.BCELoss(reduction='none')
        temp = torch.abs(y_is_box_label)
        
        ce_loss = ce(y_pred , y_is_box_label.clamp(min=0))
        loss = ( temp * ce_loss ).view(b,-1).sum(1)  / (epsilon + temp.view(b,-1).sum(1))
        return lambda_rpn_class * loss.mean()



def class_loss_cls(y_true, y_pred , lambda_cls_class= 1.0 , epsilon=1e-4):
    y_true = y_true.float()
    log_probs = (y_pred + epsilon ).log()
    num_classes  = y_true.size(1)
    # label smoothing 
    y_true = (1 - epsilon) * y_true + epsilon / num_classes
    loss = (- y_true * log_probs).mean(0).sum()
    return lambda_cls_class * loss



def class_loss_regr(y_true, y_pred , epsilon=1e-6 , lambda_cls_regr= 1.0):
    """Loss function for rpn regression
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """    
    num_classes = y_pred.size(1) // 4
    x_abs = y_true[ :, 4*num_classes:] - y_pred
    x_abs = x_abs.abs()

    index_small = torch.where(x_abs <= 1 )
    x_abs[index_small] = torch.pow(x_abs[index_small], 2) /  2 + 0.5
    x_abs = x_abs - 0.5

    loss = (y_true[ :, :4*num_classes] * x_abs).sum() / (epsilon + y_true[ :, :4*num_classes].sum() )

    return lambda_cls_regr * loss

        


