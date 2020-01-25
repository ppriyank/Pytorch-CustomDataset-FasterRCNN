# Pytorch-CustomDataset-FasterRCNN
Attempt to build Pytorch based FasterRCNN for custom dataset  ,

[PAPER](https://arxiv.org/pdf/1506.01497.pdf)
``` 
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
```


### Todo :
- [ ] classification layer
- [ ] Dataloader 
- [ ] RoiPoolingConv
- [ ] saving model and loading model
- [ ] complete calc_rpn 
- [ ] image transformation with respect to bounding box 
- [ ] get_anchor_gt
- [ ] loss functions 
- [ ] non_max_suppression_fast
- [ ] credits 
- [ ] rpn_to_roi
- [ ] training code 
- [ ] parallelize the code ?? : calc_rpn
- [ ] valid boxes
- [ ] create parsing json read/write functions
- [ ] loss functions 
- [ ] How to run and what changes to make ?? 



Tutorial / Credits / Source :  [source](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a)  ||  [Github](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras)


## Github GUIDE

* Update label_dict in main.py (assign indices to your custom classes)


## Explanation 

### Example of how aspect ratio works (Equation : 1):  
<img src="https://github.com/ppriyank/Pytorch-CustomDataset-FasterRCNN/blob/master/images/ratiologic.jpg" width="900">


### Dimension Convention :  
<img src="https://github.com/ppriyank/Pytorch-CustomDataset-FasterRCNN/blob/master/images/convention.jpg" width="900">


### anchor box

For each point on output layer of Resnet / base model (known as anchor point : 2048 , 20, 10 ==> (20 x 10) anchor points)  is the center for an anchor box on the original image. Each point has 3 bounind boxes (by default) and each of these bounding boxes comes in to different aspect ratio (3 by default) (keeping the area same, see first diagram on how aspect ratio works). Hence 9 anchor boxes are possible corresponding to each anchor points. Each of these anchor boxes (9 * 20 * 10 ) are compared with golden bounding boxes or ground truth boxes, result is either positive : (overlap with golden bounding box > 0.7) or negative (overlap with ground truth box < 0.3) or neutral (0.3 < iou < 0.7). 

Each of these anchor boxes are given label:   
* +1 for positive anchor box, 
* -1 for negative anchor box, 
* 0 for neutral (ambigious) anchor box  


For each ground truth bounding box, highest iou anchor box is kept , along with **tx, ty, th, tw** (see figure below)
<img src="https://github.com/ppriyank/Pytorch-CustomDataset-FasterRCNN/blob/master/images/bounding_box_explain.jpg" width="900">



`num_anchors_for_bbox` stores the number of positive anchor boxes associated with that golden bounding boxes.


How anchor boxes look like : 
<img src="https://github.com/ppriyank/Pytorch-CustomDataset-FasterRCNN/blob/master/images/Samples.jpg" width="900">
