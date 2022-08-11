import sys, os
import numpy as np
import cv2
from mrcnn import visualize
ROOT_DIR = "C:\\Users\\ramir\\Repos\\Mask_RCNN-akt\\"

# Import Mask RCNN
sys.path.append(ROOT_DIR) 

from custom import CustomConfig, CustomDataset
import mrcnn.model as modellib
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.8
    

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

#model_path = '.\mask_rcnn_object_0070.h5'
model_path = model.find_last()
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


import skimage
real_test_dir = './test_images/'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))
        print(filename)



def get_masked_image(image, result):
    """
    Applies masks from the results to the given image
    
    """

    boxes = result['rois']
    masks = result['masks']
    
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    colors = visualize.random_colors(N)
    masked_image = image.astype(np.uint32).copy()
    #print(colors)
    for i in range(N):
        color = colors[i]
        color = (0.0,1.0,1.0)
        # Mask
        mask = masks[:, :, i]
        
        bypass = False
        for value in boxes[i]:
            if value == 0 or result['class_ids'][i] != 1:
                bypass = True
        if not bypass:
            masked_image = visualize.apply_mask(masked_image, mask, color)
            
    return masked_image.astype(np.uint8)


for image_path in image_paths:
    

    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=0)
    print(results[0]['rois'])
    #r = results[0]

    for i, r in enumerate(results[0]):
        #seg_map = combine_masks(frame, r)
        #seg_image = label_to_color_image(seg_map)
        #frame = merge_images(seg_image, frame)
        frame = get_masked_image(img_arr, results[0])


    # image = cv2.imread(image_path)
    
    for i, roi in enumerate(results[0]['rois']):
        if results[0]['class_ids'][i] == 1:
            cv2.rectangle(frame, (roi[3], roi[2]), (roi[1], roi[0]), (255,255,255), 2 )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', frame)
    cv2.waitKey(0)

    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
    #                             ['BG', 'Lens', 'Rim'], r['scores'], figsize=(13,13))



