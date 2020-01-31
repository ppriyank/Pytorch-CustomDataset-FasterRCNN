import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image , ImageEnhance
import torchvision.transforms as transforms
from plot import verify

import random 
from tools import RPM 

class Dataset(Dataset):
    
    def __init__(self, data_folder , rpm, split, std_scaling=4.0, image_resize_size=None , debug=False):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        
        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)
        # self.labels = labels

        if self.split == 'TRAIN':
            self.transform = Transform(train=True , resize_size=image_resize_size)
        else:
            self.transform = Transform(train=False , resize_size=image_resize_size)

        self.rpm = rpm

        self.std_scaling = std_scaling
        self.image_resize_size = image_resize_size
        self.debug = debug

    def __getitem__(self, i, verify_image=False, data_format='bg_first' ):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        objects = self.objects[i]
        boxes = objects['boxes']
        labels = objects['labels']
        if data_format ==  'bg_first':
            labels = [l-1 for l in labels ]

        if verify_image:
            verify(image, boxes, labels)

        # Apply transformations
        image, boxes = self.transform.apply_transform(image, boxes)

        if self.image_resize_size: 
            y_is_box_label, y_rpn_regr, num_pos  = self.rpm.calc_rpn(boxes , labels,  image_resize_size=self.image_resize_size)
        else:
            y_is_box_label, y_rpn_regr, num_pos = self.rpm.calc_rpn(boxes , labels,  image_resize_size=(image.size[1], image.size[0] ))

        y_rpn_regr = y_rpn_regr * self.std_scaling

        if not self.debug:
            boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
            # labels = torch.LongTensor(labels)  # (n_objects)
            y_is_box_label = torch.FloatTensor(y_is_box_label)
            y_rpn_regr = torch.FloatTensor(y_rpn_regr)

        if self.debug:
            image = self.transform.to_tensor(image) 
        else:
            image = self.transform.normalize( self.transform.to_tensor(image) )
        

        return image, boxes, labels , [y_is_box_label, y_rpn_regr], num_pos

    def __len__(self):
        return len(self.images)



def collate_fn( batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels
    """

    images = list()
    boxes = list()
    labels = list()
    y_is_box_label = list()
    num_pos =  list()
    y_rpn_regr = list()


    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        y_is_box_label.append(b[3][0])
        y_rpn_regr.append(b[3][1])
        num_pos.append(b[4])
        
    images = torch.stack(images, dim=0)
    y_is_box_label = torch.cat(y_is_box_label, dim=0)
    y_rpn_regr = torch.cat(y_rpn_regr, dim=0)

    return images, boxes, labels , [y_is_box_label, y_rpn_regr] , num_pos






# dataformat : ith index ==> img_data == dict, keys : 'bboxes' , 'image' , 'class', len(dict[bboxes] == len(dict[bboxes])
def flip(image, boxes):
    # Flip image
    new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # new_image = FT.hflip(image)
    # Flip boxes
    boxes = [ [image.width - cord -1 if i % 2 ==0 else cord for i,cord in enumerate(box)  ] for box in boxes]
    boxes = [ [box[2] ,box[1] , box[0], box[3]] for box in boxes]
    return new_image, boxes



class Transform(object):
    """docstring for Transform"""
    def __init__(self,  train , resize_size=None):
        super(Transform, self).__init__()
        self.train = train 
        self.to_tensor = transforms.ToTensor()
        if resize_size:
            self.resize_size = (resize_size[1] , resize_size[0])
        else:
           self.resize_size = None 
            
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def apply_transform(self, image, boxes  ) :
        if self.resize_size:
            orig_size = image.size
            # (4032, 3024) :: w x h 

            image = image.resize( self.resize_size )
            # self.resize_size :: h x w 
            boxes= [ [cord * self.resize_size[i % 2] / orig_size[i % 2] for i,cord in enumerate(box) ] for box in boxes]

        if self.train : 
            if random.random() < 0.5:
                image , boxes = flip(image, boxes)
          
            if random.random() < 0.5:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1/8)
            
            if random.random() < 0.5:
                factor  = random.random() 
                if factor > 0.5: 
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(factor)

            if random.random() < 0.5:
                factor  = random.random() 
                if factor > 0.5: 
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(factor)

            if random.random() < 0.5:
                factor  = random.random() 
                if factor > 0.5: 
                    enhancer = ImageEnhance.Color(image)
                    image = enhancer.enhance(factor)
                
        return image , boxes 
        


class Dataset_roi(Dataset):
    
    def __init__(self, pos , neg):
        self.pos = pos
        self.neg = neg
        self.curr = -1

    def __getitem__(self, i):
        if self.pos.size(0) == 0: 
            return  [] , self.neg[i] 

        elif self.neg.size(0) == 0 : 
            return self.pos[i] , []

        else:
            if min(self.pos.size(0) , self.neg.size(0)) == self.pos.size(0):
                self.curr += 1
                return self.pos[self.curr % self.pos.size(0)] , self.neg[i]

            else:
                self.curr += 1
                return self.pos[i] , self.neg[ self.curr % self.neg.size(0) ]

    def __len__(self):
        return max(self.pos.size(0), self.neg.size(0))



