
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
