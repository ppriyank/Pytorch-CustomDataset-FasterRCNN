import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image , ImageEnhance
import torchvision.transforms as transforms
from plot import verify

image = Image.open("../IMG_0504.jpg", mode='r')




objects = {"boxes": [[1335, 1387, 1549, 1622], [1780, 1373, 1945, 1578], [2199, 1432, 2379, 1633], [2415, 1482, 2590, 1687], [2705, 1643, 2950, 1913], [3180, 1578, 3450, 1833], [3760, 1247, 4030, 1518], [3145, 1307, 3395, 1522], [3120, 1218, 3275, 1382], [2824, 1327, 3029, 1467], [2420, 1287, 2639, 1432], [2529, 1228, 2674, 1412], [2920, 1183, 3075, 1347], [3455, 1192, 3550, 1262], [3465, 1237, 3655, 1432], [0, 1738, 205, 1955], [1045, 1627, 1435, 1842], [2180, 2077, 2690, 2357], [2149, 1707, 2444, 1872], [1850, 1623, 2140, 1808], [1734, 1603, 1895, 1753], [1620, 1597, 1745, 1697], [2959, 1487, 3249, 1688], [3450, 1537, 3795, 1722], [2744, 1317, 2839, 1463], [3709, 1167, 3874, 1438], [1644, 122, 2130, 582], [2225, 393, 2684, 578], [2724, 472, 3119, 618], [3289, 283, 3824, 502], [2769, 192, 3314, 407], [3650, 9, 4031, 197], [2074, 37, 2769, 283], [699, 177, 1334, 422], [0, 68, 240, 293], [988, 0, 1699, 152]], "labels": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3], "difficulties": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}



boxes = objects['boxes']
labels = objects['labels']




class config(object):
    """docstring for config"""
    def __init__(self):
        super(config, self).__init__()
        self.voc_labels = ('laptop', 'person', 'lights', 'drinks' , 'projector')
        self.label_map = {k: v for v, k in enumerate(self.voc_labels)}
        self.label_map['bg'] = len(self.label_map)
        self.rev_label_map = {v: k for k, v in self.label_map.items()}  # Inverse mapping
        # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        self.distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

        




from tools import * 

height = 3024
width = 4032

c = config()
verify(image, boxes, labels,c)

from dataset import Transform
transform = Transform(train=False , resize_size=(height ,width))

# Apply transformations
image, boxes = transform.apply_transform(image, boxes)
verify(image, boxes, labels,c)


out_h , out_w = base_size_calculator (height  , width)

# see convention figure 
downscale = max( 
        math.ceil(height / out_h) , 
        math.ceil(width / out_w)
        )

anchor_ratios = [1,0.5,2]

min_dim = min(height, width)
index = math.floor(math.log(min_dim) /  math.log(2))
anchor_sizes = [ 2 ** index , 2 ** (index-1) , 2 ** (index-2)]

valid_anchors = valid_anchors(anchor_sizes,anchor_ratios , downscale , out_w , width , out_h , height)




temp_boxes = [list(b)[:4] for b in valid_anchors[2][1]]
temp_labels = [0 for i in range(len(temp_boxes))]

verify(image, temp_boxes, temp_labels,c)


rpm = RPM(anchor_sizes , anchor_ratios, valid_anchors, c.rev_label_map)

