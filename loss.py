import torch 
from utils import tile 

def rpn_loss_regr(y_true, y_pred , y_is_box_label , lambda_rpn_regr = 1.0):
        # Smooth L1 loss function 
        #                    0.5*x*x (if x_abs < 1)
        #                    x_abx - 0.5 (otherwise)
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
        
        
