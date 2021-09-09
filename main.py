import json
from polygon_converter import create_sub_masks
from polygon_converter import create_sub_mask_annotation
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
from PIL import Image


if __name__ == '__main__':
    # list for annotations to dump
    annotation_list = []
    images_list = []
    instance_id = 0
    for i in range(24501, 25000+1):
        print("processing image: "+str(i))
        img = Image.open('/media/hdd1/Public_Datasets/Synscapes/synscapes/Synscapes/img/instance/'+str(i)+'.png')
        with open('/media/hdd1/Public_Datasets/Synscapes/synscapes/Synscapes/meta/'+str(i)+'.json') as jsonFile:
            jsonMeta = json.load(jsonFile)
            jsonFile.close()

        instance = jsonMeta['instance']
        classes = instance['class']
        occluded = instance['occluded']

        rgb_list = []
        for id, class_id in classes.items():
            # create list to filter only instances of class 24 (person) and less than some occlusion threshold
            if class_id == 24 and occluded[id] <= 0.50:
                rgb_list.append((int(id) & 255,int(id) >> 8 & 255,int(id) >> 16))

        # create submasks of image i
        image_submasks = create_sub_masks(img, rgb_list)

        # create for each submask a polygon annotation
        for submask in image_submasks.values():
            annotation = create_sub_mask_annotation(submask, i, 1, instance_id, 0)
            if len(annotation["segmentation"]) != 0:
                annotation_list.append(annotation)
                instance_id = instance_id + 1

        width, height = img.size
        image = {"id": i, "width": width, "height": height, "file_name": str(i)+".png"}
        images_list.append(image)

    print("start creating json")
    json_data = {
        "images": images_list,
        "categories": [{"id": 1, "name": "person"}],
        "annotations": annotation_list
    }
    print("start dumping json")
    with open('my_annotation.json', 'w') as jsonFile:
        json.dump(json_data, jsonFile)
        jsonFile.close()

    print("Created successfully annotation for " + str(instance_id) + " instances")
    print("done!")

    # (R, G, B)
    # (int(id) & 255, (int(id) >> 8) & 255),int(id) >> 16)
    # #rgb = '('+ str(int(id) & 255) + ',' + str((int(id) >> 8) & 255) + ',' + str(int(id) >> 16) +')'











