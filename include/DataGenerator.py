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
import time
        

class DataGenerator(Sequence):
    def __init__(self, input_img_paths, target_ann_paths, batch_size=32, input_img_size=(640,480), target_img_size=(640,224), shuffle=True, n_channels=9, transform=None, augmentation=False):
        self.batch_size = batch_size
        self.target_ann_paths = target_ann_paths
        self.input_img_paths = input_img_paths
        self.input_img_size = input_img_size
        self.target_img_size = target_img_size
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.transform = transform
        self.augmentation = augmentation

        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def preprocess(self,img):
        height, _, _ = img.shape
        offset_y = int(self.input_img_size[0]/2)-self.target_img_size[0]
        img = img[int(height/2)+offset_y:,:,:]
        img = img.astype(np.float32)/255.0
        return img
        
    def preprocess_gray(self,img):
        height, _ = img.shape
        offset_y = int(self.input_img_size[0]/2)-self.target_img_size[0]
        img = img[int(height/2)+offset_y:,:]
        return img
    
    def decode_mask(self,mask):
        masks = np.zeros(self.target_img_size + (self.n_channels,), dtype="float32")
        for channel_i in range(1,self.n_channels):
            channel_mask = np.where(mask == channel_i*20, 1.0, 0.0)
            masks[:,:,channel_i-1] = channel_mask
        # background channel
        masks[:,:,self.n_channels-1] = np.where(mask == 0, 1.0,0.0)
        return masks
            
    def data_generation(self, batch_input_img_path, batch_target_ann_path):
        start_b = time.time()
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,) + self.target_img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.target_img_size + (self.n_channels,), dtype="float32")
        
        # Generate data
        for j, img_path in enumerate(batch_input_img_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(batch_target_ann_path[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask =cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            if self.augmentation and self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            
            img = self.preprocess(img)
            mask = self.preprocess_gray(mask)
            masks = self.decode_mask(mask)

            X[j] = img
            y[j] = masks
        
        print(f"whole batch took: {(time.time()-start_b)*1000}ms")
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