import os 
import torchvision.transforms as transforms 
import torch
import copy 

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




def verify2(image, boxes, labels, config , color, name="", plot_labels=True):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 15)
    for i in range( len(boxes) ):
        # Boxes
        box_location = boxes[i]
        draw.rectangle(xy=box_location, outline= color )
        draw.rectangle(xy=[l + 1. for l in box_location], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness
        # Text
        if plot_labels :
            text_size = font.getsize(labels)
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill= color )
            draw.text(xy=text_location, text=labels, fill='white',font=font)
    image.show()
    image.save(name + ".jpg") 



def save_evaluations_image(image, boxes, labels, count, config , save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + "pictures/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    

    trans = transforms.ToPILImage()
    image = trans(image.cpu())
    image.save(save_dir + str(count) + ".jpg") 
    pos_image=  copy.deepcopy(image)  

    neg_samples = torch.where(labels[:, -1] == 1)[0]
    pos_samples = torch.where(labels[:, -1] == 0)[0]
    
    # if args.data_format == 'bg_first':
    #   labels = torch.cat([labels[:,-1].unsqueeze(1) , labels[:,:-1]] , 1)
    #   labels = torch.cat([ labels[:,1:] , labels[:,0].unsqueeze(1)] , 1)

    labels = labels.cpu() 
    boxes = boxes.cpu()

    draw_pos = ImageDraw.Draw(pos_image)
    draw_neg = ImageDraw.Draw(image)

    font = ImageFont.truetype("arial.ttf", 15)
    for i in range( boxes.size(0) ):
        # Boxes
        ind = int(torch.where(labels[i] == 1)[0].numpy() )
        box_location = boxes[i]
        config 
        if i in neg_samples:
            draw = draw_neg
            color = 'red'
        else:
            draw = draw_pos
            color = 'green'
        draw.rectangle(xy=box_location.numpy(), outline= color )
        draw.rectangle(xy=[l + 1. for l in box_location.numpy()], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness
        # Text
        
        name = config.rev_label_map[ind]
        text_size = font.getsize(name)
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill= color )
        draw.text(xy=text_location, text=name, fill='white',font=font)

    pos_image.save(save_dir + str(count) + "_pos" + ".jpg") 
    image.save(save_dir + str(count) + "_neg" + ".jpg") 
    # image.show()
    





