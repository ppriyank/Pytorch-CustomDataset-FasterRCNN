import math 
import numpy as np 
import random 
from utils import iou

# Code taken from 
# https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/frcnn_train_vgg.ipynb


def base_size_calculator(h,w):
	# FOR RESNET
	# output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
	output_size_conv = int((h - 7 + 2 * 3 )/ 2 ) + 1 , int((w - 7 + 2 * 3 )/ 2 ) + 1
	output_size_maxpool = int((output_size_conv[0] - 3 + 2 * 1 )/ 2 ) + 1 , int((output_size_conv[1] - 3 + 2 * 1 )/ 2 ) + 1

	output_size_layer1 = int((output_size_maxpool[0] - 1  )/ 1 ) + 1 , int((output_size_maxpool[1] - 1 )/ 1 ) + 1
	output_size_layer2 = int((output_size_layer1[0] - 3  + 2)/ 2 ) + 1 , int((output_size_layer1[1] - 3 + 2 )/ 2 ) + 1
	output_size_layer3 = int((output_size_layer2[0] - 3  + 2)/ 2 ) + 1 , int((output_size_layer2[1] - 3 + 2 )/ 2 ) + 1
	output_size_layer4 = output_size_layer3

	return output_size_layer4




def valid_anchors(anchor_sizes,anchor_ratios , downscale , output_width , resized_width , output_height , resized_height):
	anchor_boxes = {}
	n_anchratios = len(anchor_ratios) # 3
	for anchor_size_idx in range(len(anchor_sizes)):
		anchor_boxes[anchor_size_idx] = {}
		for anchor_ratio_idx in range(n_anchratios):
			anchor_boxes[anchor_size_idx][anchor_ratio_idx] = []
			# (Equation : 1, see figure)
			ratio_square_root = abs(math.sqrt(anchor_ratios[anchor_ratio_idx]))
			anchor_x = anchor_sizes[anchor_size_idx] * ratio_square_root
			anchor_y = anchor_sizes[anchor_size_idx] / ratio_square_root
			for ix in range(output_width):		
				# x-coordinates of the current anchor box	
				# x1_anc = downscale * (ix ) - anchor_x / 2
				# x2_anc = downscale * (ix ) + anchor_x / 2	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_width:
					continue
				for jy in range(output_height):
					# y-coordinates of the current anchor box
					# y1_anc = downscale * (jy ) - anchor_y / 2
					# y2_anc = downscale * (jy ) + anchor_y / 2
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2
					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue
					anchor_boxes[anchor_size_idx][anchor_ratio_idx].append((x1_anc , y1_anc , x2_anc,  y2_anc , ix , jy))
	return anchor_boxes
		




class RPM():
	def __init__(self, anchor_sizes , anchor_ratios, valid_anchors, rev_label_map, rpn_max_overlap=0.7 , rpn_min_overlap=0.3, num_regions = 500 ):
		super(RPM, self).__init__()
		self.anchor_sizes = anchor_sizes
		self.anchor_ratios = anchor_ratios
		self.valid_anchors = valid_anchors
		self.rpn_max_overlap = rpn_max_overlap
		self.rpn_min_overlap = rpn_min_overlap
		self.rev_label_map = rev_label_map
		self.num_regions = num_regions
		
	
	def calc_rpn(self, boxes , labels , image_resize_size=(300,400) ): 
		num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios) # 3x3=9
		n_anchratios = len(self.anchor_ratios) # 3
		(output_height , output_width) = base_size_calculator(image_resize_size[0], image_resize_size[1])

		y_is_box_label = np.zeros((output_height, output_width, num_anchors))
		y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))
		
		num_bboxes = len(boxes)

		num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
		best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
		best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
		best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
		best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

		gta = np.array((boxes))

		for key1 in self.valid_anchors:
			for key2 in self.valid_anchors[key1]:
				for anchor_box in self.valid_anchors[key1][key2]: 
					anchor_ratio_idx = key2
					anchor_size_idx = key1

					x1_anc , y1_anc , x2_anc , y2_anc , ix , jy = anchor_box
					# bbox_type indicates whether an anchor should be a target
					# Initialize with 'negative'
					bbox_type = 'neg'
					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0
					# get IOU of the current GT box and the current anchor box
					for bbox_num in range(num_bboxes):
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# import pdb
						# pdb.set_trace()

						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num]  or curr_iou > self.rpn_max_overlap : 
							
							golden_center_x = (gta[bbox_num, 0] + gta[bbox_num, 2]) / 2.0
							golden_center_y = (gta[bbox_num, 1] + gta[bbox_num, 3]) / 2.0

							anchor_center_x = (x1_anc + x2_anc)/2.0
							anchor_center_y = (y1_anc + y2_anc)/2.0
							
							tx = (golden_center_x - anchor_center_x) / (x2_anc - x1_anc)
							ty = (golden_center_y - anchor_center_y) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 2] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 1]) / (y2_anc - y1_anc))

						
						if self.rev_label_map[labels[bbox_num]] != 'bg':	
							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, y1_anc , x2_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > self.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if self.rpn_min_overlap < curr_iou and curr_iou < self.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'


					# turn on or off outputs depending on IOUs
					if bbox_type == 'neg':
						y_is_box_label[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = -1
					elif bbox_type == 'neutral':
						y_is_box_label[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_label[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr

							
		# we ensure that every bbox has at least one positive RPN region
		for idx in range(num_anchors_for_bbox.shape[0]):
			if num_anchors_for_bbox[idx] == 0:
				# no box with an IOU greater than zero ...
				if best_anchor_for_bbox[idx, 0] == -1:
					continue
				y_is_box_label[
				best_anchor_for_bbox[idx,0], 
				best_anchor_for_bbox[idx,1], 
				best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3]
				] = 1
				start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
				y_rpn_regr[
					best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]


		
		pos_locs = np.where(y_is_box_label == 1 )
		num_pos = len(pos_locs[0])

		neg_locs = np.where(y_is_box_label == -1 )
		num_neg =  len(neg_locs[0])

		# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
		# regions. We also limit it to 256 regions.
		if num_pos > self.num_regions/2:
			non_valid_boxes  = random.sample(range( num_pos ), num_pos - self.num_regions/2)
			y_is_box_label[pos_locs[0][non_valid_boxes], pos_locs[1][non_valid_boxes], pos_locs[2][non_valid_boxes]] = 0 
			num_pos = self.num_regions/2

		if num_neg + num_pos > self.num_regions:
			non_valid_boxes = random.sample(range(num_neg), num_neg - num_pos)
			y_is_box_label[neg_locs[0][non_valid_boxes], neg_locs[1][non_valid_boxes], neg_locs[2][non_valid_boxes]] = 0 

		# y_is_box_label = np.transpose(y_is_box_label, (2, 0, 1))
		y_is_box_label = np.expand_dims(y_is_box_label, axis=0)

		# y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
		y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
		
		# y_rpn_regr = np.concatenate([np.repeat(y_is_box_label, 4, axis=1), y_rpn_regr], axis=1)

		return y_is_box_label, y_rpn_regr, num_pos
		# return np.copy(y_is_box_label), np.copy(y_rpn_regr), num_pos


 		













# Code taken from here: 
# https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/frcnn_train_vgg.ipynb
# Reduce the overlapping boxes to 1 


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=500):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []
    # calculate the areas
    area = (x2 - x1) * (y2 - y1)
    
    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

def apply_regr_np(X, T):
    """Apply regression layer to all anchors in one feature map

    Args:
        X: shape=(4, 18, 25) the current anchor type for all points in the feature map
        T: regression layer shape=(4, 18, 25)

    Returns:
        X: regressed position and size for current anchor
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X
    
def apply_regr(x, y, w, h, tx, ty, tw, th):
    # Apply regression to x, y, w and h
    try:
        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h

def calc_iou(R, img_data, C, class_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        R: bboxes, probs
    """
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        # gta[bbox_num, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    # R.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # Iterate through all the ground-truth bboxes to calculate the iou
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])

            # Find out the corresponding ground-truth bbox_num with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot code for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs






def rpn_to_roi(cls_k, reg_k, anchor_sizes , anchor_ratios , use_regr=True, max_boxes=500, overlap_thresh=0.9 , std_scaling=4.0 ):
	"""
	Returns:
		result: boxes from non-max-suppression (shape=(max_boxes, 4))
			boxes: coordinates for bboxes (on the feature map)
	"""
	reg_k = reg_k / std_scaling

	rows = cls_k.size(1)
	cols = cls_k.size(2)
	curr_layer = 0

	# A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
	# Might be (4, 18, 25, 18) if resized image is 400 width and 300
	# A is the coordinates for 9 anchors for every point in the feature map 
	# => all 18x25x9=4050 anchors cooridnates
	A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:
			# anchor_x = (128 * 1) / 16 = 8  => width of current anchor
			# anchor_y = (128 * 2) / 16 = 16 => height of current anchor
			anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
			anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
			
			# curr_layer: 0~8 (9 anchors)
			# the Kth anchor of all position in the feature map (9th in total)
			regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4] # shape => (18, 25, 4)
			regr = np.transpose(regr, (2, 0, 1)) # shape => (4, 18, 25)

			# Create 18x25 mesh grid
			# For every point in x, there are all the y points and vice versa
			# X.shape = (18, 25)
			# Y.shape = (18, 25)
			X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

			# Calculate anchor position and size for each feature map point
			A[0, :, :, curr_layer] = X - anchor_x/2 # Top left x coordinate
			A[1, :, :, curr_layer] = Y - anchor_y/2 # Top left y coordinate
			A[2, :, :, curr_layer] = anchor_x       # width of current anchor
			A[3, :, :, curr_layer] = anchor_y       # height of current anchor

			# Apply regression to x, y, w and h if there is rpn regression layer
			if use_regr:
				A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

			# Avoid width and height exceeding 1
			A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

			# Convert (x, y , w, h) to (x1, y1, x2, y2)
			# x1, y1 is top left coordinate
			# x2, y2 is bottom right coordinate
			A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

			# Avoid bboxes drawn outside the feature map
			A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
			A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
			A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4050, 4)
	all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))                   # shape=(4050,)

	x1 = all_boxes[:, 0]
	y1 = all_boxes[:, 1]
	x2 = all_boxes[:, 2]
	y2 = all_boxes[:, 3]

	# Find out the bboxes which is illegal and delete them from bboxes list
	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)

	# Apply non_max_suppression
	# Only extract the bboxes. Don't need rpn probs in the later process
	result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

	return result



