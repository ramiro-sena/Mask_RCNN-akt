import sys, os
import numpy as np
from mrcnn import visualize
import cv2
ROOT_DIR = "C:\\Users\\ramir\\Repos\\Mask_RCNN-master"

# Import Mask RCNN
from mrcnn.config import Config
import mrcnn.model as modellib

sys.path.append(ROOT_DIR) 
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = ROOT_DIR

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = '.\mask_rcnn_coco.h5'
#model_path = model.find_last()
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


import skimage
real_test_dir = './images/'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))
        print(filename)


for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]

    classes= []
    for c in range(0,81):
        classes.append(f'class {c}')
    print(r['rois'])
    image = cv2.imread(image_path)
    
    for roi in r['rois']:
        cv2.rectangle(image, (roi[3], roi[2]), (roi[1], roi[0]), (255,0,0), 1 )
    cv2.imshow('img', image)
    cv2.waitKey(0)
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
    #                             classes, r['scores'], figsize=(5,5))