import sys, os
import cv2
import numpy as np
from mrcnn import visualize
ROOT_DIR = "C:\\Users\\ramir\\Repos\\Mask_RCNN-akt\\"

# Import Mask RCNN
sys.path.append(ROOT_DIR) 

from custom import CustomConfig, CustomDataset
import mrcnn.model as modellib
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    IMAGE_MIN_DIM = 240
    IMAGE_MAX_DIM = 1024
    # DETECTION_MIN_CONFIDENCE = 0.8
    

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

#model_path = '.\mask_rcnn_object_0070.h5'
model_path = model.find_last()
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", 'logs\object20220810T0246\mask_rcnn_object_0048.h5')
model.load_weights(model_path, by_name=True)


VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")

try:
    if not os.path.exists(VIDEO_SAVE_DIR):
        os.makedirs(VIDEO_SAVE_DIR)
except OSError:
    print ('Error: Creating directory of data')


# number of images to be processed at once
batch_size = inference_config.IMAGES_PER_GPU
capture = cv2.VideoCapture("C:\\Users\\ramir\\Videos\\online-prescription-glasses-2019-0246.mp4")


try:
    if not os.path.exists(VIDEO_SAVE_DIR):
        os.makedirs(VIDEO_SAVE_DIR)
except OSError:
    print ('Error: Creating directory of data')
frames = []
frame_count = 0

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
        masked_image = visualize.apply_mask(masked_image, mask, color)
    return masked_image.astype(np.uint8)


while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break
        
    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)
    print('frame_count :{0}'.format(frame_count))
    if len(frames) == batch_size:
        results = model.detect(frames, verbose=0)
        print('Predicted')
        for i, item in enumerate(zip(frames, results)):
            frame = item[0]
            r = item[1]
            #seg_map = combine_masks(frame, r)
            #seg_image = label_to_color_image(seg_map)
            #frame = merge_images(seg_image, frame)
            frame = get_masked_image(frame, r)
            name = '{0}.jpg'.format(frame_count + i - batch_size)
            name = os.path.join(VIDEO_SAVE_DIR, name)
            cv2.imwrite(name, frame)
        # Clear the frames array to start the next batch
        frames = []



def make_video(outvid, images=None, fps=30, size=None,
            is_color=True, format="FMP4"):


    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


import glob
import os

images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = os.path.join(VIDEO_DIR, "final.mp4")
make_video(outvid, images, fps=20)