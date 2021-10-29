import json
from os.path import join
import pickle 

def read_all(directory, filename):
    with open(join(directory, filename), 'r') as f:
        lines = f.readlines()
    return json.loads(lines[0])



directory = "/nfs/bigcornea/add_disk0/pathak/biodata2/BBBC041"
image_directory=  join(directory, 'images')

data_train = read_all(directory, "training.json")
data_test = read_all(directory, "test.json")

voc_labels = ('schizont', 'gametocyte', 'trophozoite', 'red blood cell', 'difficult', 'ring', 'leukocyte')
voc_labels += ('bg',)
label_map = {k: v for v, k in enumerate(voc_labels)}

def process(data):
    image_paths = []
    labels = []
    for e in data:
        path = directory + e["image"]["pathname"]
        image_paths.append(path)
        boxes = {"boxes": [] , "labels" : []}
        for elements in e["objects"]:
            bounding_box , label = elements["bounding_box"] , elements["category"]
            min_y = bounding_box['minimum']['r']
            min_x = bounding_box['minimum']['c']
            max_y = bounding_box['maximum']['r']
            max_x = bounding_box['maximum']['c']
            boxes["boxes"].append([min_x, min_y, max_x, max_y])
            boxes["labels"].append(label_map[label])
        labels.append(boxes)
    return labels , image_paths



data_train = process(data_train)
data_test = process(data_test)


with open("TRAIN_images.json", 'w') as output:
	output.write( str(data_train[1]).replace("'", "\"") )


with open("TRAIN_objects.json", 'w') as output:
	output.write( str(data_train[0]).replace("'", "\"")  )


with open("TEST_images.json", 'w') as output:
	output.write(  str(data_test[1]).replace("'", "\"")  )


with open("TEST_objects.json", 'w') as output:
	output.write(  str(data_test[0]).replace("'", "\"") )


	