#!/usr/bin/env python3

import sys
import os
os.system('export SM_FRAMEWORK=tf.keras')
import cv2
import numpy as np
import fnmatch
from tensorflow.keras.models import load_model, Model
import tensorflow.keras as keras
import segmentation_models as sm

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

record = True

number_classes = 9 # outer_l, outer_t, outer_r, middle_curb
x_values = 192
y_values = 48

input_width = 434
input_height = 150

size = (868, 300)

model_dependencies = {
    'iou_score': sm.metrics.iou_score
}

def preprocess(img):
    height, _, _ = img.shape
    img = img[int(height/2):,:,:]
    return img
def postprocess_channel(img):
    thres_value = 0.3
    #img = cv2.medianBlur(img, 3)
    _, img = cv2.threshold(img,thres_value,1.0,cv2.THRESH_BINARY)
    return img

def draw_frame(img, prediction):
    test_img = img/255
    test_img = test_img.astype(np.float32)
    predicted_outer = np.sum([postprocess_channel(prediction[:,:,i]) for i in range(4)], axis=0)
    predicted_middle = np.sum([postprocess_channel(prediction[:,:,i]) for i in range(4,6)], axis=0)
    predicted_wait = np.sum([postprocess_channel(prediction[:,:,i]) for i in range(6,8)], axis=0)
    predicted_lanes = cv2.merge([predicted_outer, predicted_wait, predicted_middle])
    predicted_lanes = cv2.resize(predicted_lanes, (input_width, input_height))
    overlay_image = cv2.addWeighted(test_img, 0.4, predicted_lanes, 0.6, 0)
    overlay_image = cv2.resize(overlay_image, size)
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    overlay_image = overlay_image*255
    overlay_image = np.uint8(overlay_image)
    return overlay_image

def main_images(model, input_path, output_path):
    file_list = os.listdir(input_path)
    file_list.sort()
    pattern = '*.jpg'
    try:
        running = True
        if record:
            out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        while running: 
            for filename in file_list:
                if fnmatch.fnmatch(filename, pattern):
                    image_path = os.path.join(input_path, filename)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess(img)
                    img = cv2.resize(img, (input_width, input_height))
                    
                    prediction = model.predict(np.array([img]))[0]
                    overlay_image = draw_frame(img, prediction)
                    cv2.imshow('frame', overlay_image)
                    if record:
                        out.write(overlay_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        running = False
                        if record:
                            out.release()
                        break   
    except KeyboardInterrupt:
        print('interrupt')
    if record:
        out.release()
    cv2.destroyAllWindows()

def main_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    try:
        if record:
            out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        while cap.isOpened():
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess(img)
            img = cv2.resize(img, (input_width, input_height))
            
            prediction = model.predict(np.array([img]))[0]
            overlay_image = draw_frame(img, prediction)
            cv2.imshow('frame', overlay_image)
            if record:
                out.write(overlay_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                if record:
                    out.release()
                break
    except KeyboardInterrupt:
        print('interrupt')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = sys.argv[1]
    model = load_model(model_path, custom_objects=model_dependencies)
    input_path = sys.argv[2]
    if os.path.isfile(input_path):   
        output_path = f"{os.path.splitext(input_path)[0]}_demo.mp4"
        main_video(model, input_path, output_path)
    else:
        output_path = f"{input_path}_demo.mp4"
        main_images(model, input_path, output_path)