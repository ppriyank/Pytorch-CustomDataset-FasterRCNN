import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image , ImageEnhance
import torchvision.transforms as transforms
from plot import verify

from tools import * 

height = 3024
width = 4032

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
valid_anchors = valid_anchors(anchor_sizes,anchor_ratios , downscale , output_width=out_w , resized_width=width , output_height=out_h , resized_height=height)




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

        
c = config()

########################################################################################################################################################################################################################
# check for transformation 
########################################################################################################################################################################################################################

image = Image.open("../IMG_0504.jpg", mode='r')
objects = {"boxes": [[1335, 1387, 1549, 1622], [1780, 1373, 1945, 1578], [2199, 1432, 2379, 1633], [2415, 1482, 2590, 1687], [2705, 1643, 2950, 1913], [3180, 1578, 3450, 1833], [3760, 1247, 4030, 1518], [3145, 1307, 3395, 1522], [3120, 1218, 3275, 1382], [2824, 1327, 3029, 1467], [2420, 1287, 2639, 1432], [2529, 1228, 2674, 1412], [2920, 1183, 3075, 1347], [3455, 1192, 3550, 1262], [3465, 1237, 3655, 1432], [0, 1738, 205, 1955], [1045, 1627, 1435, 1842], [2180, 2077, 2690, 2357], [2149, 1707, 2444, 1872], [1850, 1623, 2140, 1808], [1734, 1603, 1895, 1753], [1620, 1597, 1745, 1697], [2959, 1487, 3249, 1688], [3450, 1537, 3795, 1722], [2744, 1317, 2839, 1463], [3709, 1167, 3874, 1438], [1644, 122, 2130, 582], [2225, 393, 2684, 578], [2724, 472, 3119, 618], [3289, 283, 3824, 502], [2769, 192, 3314, 407], [3650, 9, 4031, 197], [2074, 37, 2769, 283], [699, 177, 1334, 422], [0, 68, 240, 293], [988, 0, 1699, 152]], "labels": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3], "difficulties": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
boxes = objects['boxes']
labels = objects['labels']

from dataset import Transform
verify(image, boxes, labels,c)
transform = Transform(train=True , resize_size=(height ,width))
# Apply transformations
image, boxes = transform.apply_transform(image, boxes)
verify(image, boxes, labels,c)
# temp_boxes = [list(b)[:4] for b in valid_anchors[2][1]]
# temp_labels = [0 for i in range(len(temp_boxes))]
# verify(image, temp_boxes, temp_labels,c)



########################################################################################################################################################################################################################
# check dataloader 
########################################################################################################################################################################################################################

from dataset import Dataset
import torchvision.transforms as transforms
from plot import verify2 

db =  Dataset(data_folder=".", anchor_sizes = anchor_sizes, anchor_ratios = anchor_ratios, valid_anchors=valid_anchors, rev_label_map=config.rev_label_map,  split='TRAIN', debug= True)
c= next(iter(db))
Y = c[3]

cls = Y[0][0]
regr = Y[1][0]

pos_cls = np.where(cls==1)
print(pos_cls)

possible_anchors = cls[pos_cls[0],pos_cls[1],:]
possible_regr = regr[pos_cls[0],pos_cls[1],:] 

box = c[1]
labels = c[2]

trans = transforms.ToPILImage()
img = trans(c[0]).convert("RGB")
verify2(img, box, labels="GT", config= config , color='#e6194b')

debug_num_pos = c[-1]
potential = [] 

n_anchratios = len(anchor_ratios)

for i in range(debug_num_pos):
yc = pos_cls[0][i] 
xc = pos_cls[1][i]
# this will work only if len(anchor_sizes) <= n_anchratios 
anchor_size_idx = pos_cls[2][i] // n_anchratios
anchor_ratio_idx = pos_cls[2][i] - anchor_size_idx * n_anchratios 
ratio_square_root = math.sqrt(anchor_ratios[anchor_ratio_idx])
anchor_x = anchor_sizes[anchor_size_idx] / ratio_square_root
anchor_y = anchor_sizes[anchor_size_idx] * ratio_square_root
x1_anc = downscale * (xc + 0.5) - anchor_x / 2
x2_anc = downscale * (xc + 0.5) + anchor_x / 2	
y1_anc = downscale * (yc + 0.5) - anchor_y / 2
y2_anc = downscale * (yc + 0.5) + anchor_y / 2
				
				


	y_is_box_label[] = -1


		a , b, pos_cls[]
        idx = pos_regr[2][i*4]/4
        anchor_size = C.anchor_box_scales[int(idx/3)]
        anchor_ratio = C.anchor_box_ratios[2-int((idx+1)%3)]

        center = (pos_regr[1][i*4]*C.rpn_stride, pos_regr[0][i*4]*C.rpn_stride)
        print('Center position of positive anchor: ', center)
        cv2.circle(img, center, 3, color, -1)
        anc_w, anc_h = anchor_size*anchor_ratio[0], anchor_size*anchor_ratio[1]
        cv2.rectangle(img, (center[0]-int(anc_w/2), center[1]-int(anc_h/2)), (center[0]+int(anc_w/2), center[1]+int(anc_h/2)), color, 2)


verify2(img, box, labels="Potential", config= config , color='#f58231')




pos_regr = np.where(regr==1)





z.show()


import matplotlib.pyplot as plt
plt.imshow(z)


