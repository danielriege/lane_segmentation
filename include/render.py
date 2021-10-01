import numpy as np
import supervisely_parser as svp
import grid_parser as gp
import cv2
import os
import fnmatch

def render_masks(ann_path, mask_size):
    # get annotation
    lanes = svp.getPoints(ann_path)
    # draws masks into multi channel array from labled points
    segmented_data = svp.drawLanes(mask_size, lanes)
    # merge grayscale masks into multi channel image and preprocess data
    merged = cv2.merge([segmented for segmented in segmented_data])
    return merged

def render_packages(packages, ann_base, mask_size):
    for index in range(len(packages)):
        ann_base_path = f"{ann_base}{packages[index]}/ann/"
        masks_base_path = f"{ann_base}{packages[index]}/masks/"
        # create new dir for every package
        if not os.path.exists(masks_base_path):
            os.makedirs(masks_base_path)
        # for every ann file
        file_list = os.listdir(ann_base_path)
        pattern = '*.json'
        for filename in file_list:
            if fnmatch.fnmatch(filename, pattern):
                ann_file = os.path.join(ann_base_path, filename)
                # render masks
                masks = render_masks(ann_file, mask_size)
                # encode masks into 1 channel image
                grayscale_img = np.zeros(mask_size)
                for channel in range(masks.shape[2]):
                    mask = masks[:,:,channel]
                    grayscale_img += (channel+1)*20*mask
                out_path = os.path.join(masks_base_path, filename)
                size = len(out_path)
                out_path = out_path[:size-9]
                out_path = f"{out_path}.png"
                cv2.imwrite(out_path, grayscale_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])