#!/usr/bin/env python3

import numpy as np

def calc_ratio_data(data, number_classes, class_names):
    ratio_data = []
    index_data = []
    for j in range(number_classes):
        ratio_data.append([])
        index_data.append([])
    for i,masks in enumerate(data):
        for class_idx in range(number_classes):
            numbers_of_class_pixels = (masks[:,:,class_idx] > 0.5).sum()
            if numbers_of_class_pixels > 0:
                ratio_data[class_idx].append(numbers_of_class_pixels)
                index_data[class_idx].append(i)
    return (ratio_data, index_data)

def calc_chart_data(ratio_data, number_classes):
    hist_by_pixel = []
    hist_by_images = []
    for class_idx, ratio_arr in enumerate(ratio_data):
        if class_idx != number_classes-1:
            hist_by_pixel.append(sum(ratio_arr))
            hist_by_images.append(len(ratio_arr))
            
    return (hist_by_pixel, hist_by_images)