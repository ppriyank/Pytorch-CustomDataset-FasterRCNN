
from PIL import Image, ImageDraw, ImageFont
  

def verify(image, boxes, labels, c):
	label_color_map = {k: c.distinct_colors[i] for i, k in enumerate(c.rev_label_map.keys())}
	draw = ImageDraw.Draw(image)
	font = ImageFont.truetype("arial.ttf", 15)
	for i in range( len(boxes) ):
	    # Boxes
	    box_location = boxes[i]
	    draw.rectangle(xy=box_location, outline=label_color_map[labels[i]])
	    draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
	    # Text
	    text_size = font.getsize(c.rev_label_map[labels[i]].upper())
	    text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
	    textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
	                        box_location[1]]
	    draw.rectangle(xy=textbox_location, fill=label_color_map[labels[i]])
	    draw.text(xy=text_location, text=c.rev_label_map[labels[i]].upper(), fill='white',font=font)
	image.show()




def verify2(image, boxes, labels, config , color, name=""):
	draw = ImageDraw.Draw(image)
	font = ImageFont.truetype("arial.ttf", 15)
	for i in range( len(boxes) ):
	    # Boxes
	    box_location = boxes[i]
	    draw.rectangle(xy=box_location, outline= color )
	    draw.rectangle(xy=[l + 1. for l in box_location], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness
	    # Text
	    text_size = font.getsize(labels)
	    text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
	    textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
	                        box_location[1]]
	    draw.rectangle(xy=textbox_location, fill= color )
	    draw.text(xy=text_location, text=labels, fill='white',font=font)
	image.show()
	image.save(name + ".jpg") 

color = (0, 255, 0)


def true_box(image, boxes, labels,name="verify/", id=0):
	try:
	    os.mkdir(name)
	except OSError:
		None
	    
	
	annotated_image = image
	# image = normalize(to_tensor(resize(image)))
	det_boxes = boxes.to('cpu')
	# original_dims = torch.FloatTensor([300, 300, 300, 300]).unsqueeze(0)
	# det_boxes = det_boxes * original_dims
	det_labels = [rev_label_map[l] for l in labels.to('cpu').tolist()]
	# images_array = image.to('cpu').permute(1,2,0).numpy()
	# temp = (images_array - np.min(images_array)) * 255 // (np.max(images_array) - np.min(images_array))
	# annotated_image = Image.fromarray(np.array(temp).astype(np.uint8))
	original_image = annotated_image
	# original_image.save(name + "_temp2.jpg", "JPEG")
	draw = ImageDraw.Draw(annotated_image)
	font = ImageFont.truetype("./arial.ttf", 15)
	label_color_map
	for i in range(det_boxes.size(0)):
	    # Boxes
	    box_location = det_boxes[i].tolist()
	    draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
	    draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
	        det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
	    # Text
	    text_size = font.getsize(det_labels[i].upper())
	    text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
	    textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
	                        box_location[1]]
	    draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
	    draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
	              font=font)
	annotated_image.save(name + str(id) + ".jpg", "JPEG")
