import torch 
import numpy as np 

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)




def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


# def save_checkpoint(epoch, model, loss, is_best, name="_"):
#     state = {'epoch': epoch,
#              'loss': loss,
#              'model': model}
#     filename = name + 'checkpoint_ssd300.pth.tar'
#     # torch.save(state, filename)
#     # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
#     if is_best:
#         torch.save(state, 'BEST_' + filename)


# def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
#     mkdir_if_missing(osp.dirname(fpath))
#     matching_file = fpath.split("ep")[0].split("/")[-1]
#     dir_ =  osp.dirname(fpath)
#     for f in os.listdir(dir_):
#         if re.search(matching_file, f):
#             os.remove(os.path.join(dir_, f))            
#     torch.save(state, fpath)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


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

