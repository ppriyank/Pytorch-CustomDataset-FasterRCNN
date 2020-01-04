# Pytorch-CustomDataset-FasterRCNN
Attempt to build Pytorch based FasterRCNN for custom dataset  ,

[PAPER](https://arxiv.org/pdf/1506.01497.pdf)
``` 
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
```


### Todo : 
* classification layer
* Dataloader 
* RoiPoolingConv
* saving model and loading model
* complete calc_rpn 
* image transformation with respect to bounding box 
* get_anchor_gt
* loss functions 
* non_max_suppression_fast
* credits 
* rpn_to_roi
* training code 
* parallelize the code ?? : calc_rpn
* valid boxes
* create parsing json read/write functions




Tutorial / Credits / Source :  [source](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a)  ||  [Github](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras)


## Github GUIDE

* Update label_dict in main.py (assign indices to your custom classes)


## Explanation 

Example of how aspect ratio works (Equation : 1):  
<img src="https://github.com/ppriyank/Pytorch-CustomDataset-FasterRCNN/blob/master/images/ratiologic.jpg" width="900">


Dimension Convention :  
<img src="https://github.com/ppriyank/Pytorch-CustomDataset-FasterRCNN/blob/master/images/convention.jpg" width="900">

