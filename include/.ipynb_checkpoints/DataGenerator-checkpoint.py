#!/usr/bin/env python3
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
#from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
import supervisely_parser as svp
import grid_parser as gp
import cv2
import os
import fnmatch
import albumentations as Alb

class DataGenerator(Sequence):
    def __init__(self, input_img_paths, target_ann_paths, batch_size=32, input_img_size=(640,480), target_img_size=(640,224), shuffle=True, n_channels=9, transform=None):
        self.batch_size = batch_size
        self.target_ann_paths = target_ann_paths
        self.input_img_paths = input_img_paths
        self.input_img_size = input_img_size
        self.target_img_size = target_img_size
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.transform = transform
        
        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def preprocess(self,img):
        height, _, _ = img.shape
        img = img[int(height/2):,:,:]
        img = img.astype(np.float32)/255.0
        return img

    def preprocess_gray(self,img):
        height, _ = img.shape
        img = img[int(height/2):,:]
        return img
            
    def data_generation(self, batch_input_img_path, batch_target_ann_path):
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,) + (int(self.input_img_size[0]/2), self.input_img_size[1]) + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.target_img_size + (self.n_channels,), dtype="float32")
        use_augmentation = False

        # Generate data
        for j, img_path in enumerate(batch_input_img_path):
            if fnmatch.fnmatch(img_path, "*.aug"):
                use_augmentation = True
                img_path = os.path.splitext(img_path)[0]
            # get sample
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # get annotation
            ann_path = batch_target_ann_path[j]
            lanes = svp.getPoints(ann_path)
            # draws masks into multi channel array from labled points
            segmented_data = svp.drawLanes(self.input_img_size, lanes)
            # apply augmentation to both images when used
            if use_augmentation and self.transform is not None:
                transformed = self.transform(image=img, masks=segmented_data)
                img = transformed['image']
                segmented_data = np.array(transformed['masks'])
            # after optional augmentation, preprocess both images to desired sizes and crops
            # merge grayscale masks into multi channel image and preprocess data
            merged = cv2.merge([cv2.resize(self.preprocess_gray(segmented), (self.target_img_size[1],self.target_img_size[0])) for segmented in segmented_data])
            
            img = self.preprocess(img)
            
            X[j] = img
            y[j] = merged

        return X, y
    
    def __len__(self):
        return int(np.floor(len(self.input_img_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of input and target path
        batch_input_img_path = [self.input_img_paths[k] for k in indexes]
        batch_target_ann_path = [self.target_ann_paths[k] for k in indexes]

        # Generate data
        x, y = self.data_generation(batch_input_img_path, batch_target_ann_path)

        return x, y