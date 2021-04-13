#!/usr/bin/env python3

import numpy as np
import cv2
def foo(l, dtype=int):
    return list(map(dtype, l))

def segmented_image_into_grid_space(image, image_y_range=None, grid_size = (10, 36)):
    '''
    w = grid_size[1] + 1 which is number of row anchors
    the plus one is the class for lane not on image
    '''
    if image_y_range == None:
        image_y_range = (0, image.shape[0])
    w = grid_size[1]+1
    window_size = np.int((image_y_range[1]-image_y_range[0])/grid_size[0])
    window_start = image_y_range[0]
    grid_data = []
    for channel_index in range(image.shape[2]):
        channel_image = image[:,:,channel_index]
        grid_lane_x = np.full(grid_size[0], int(w))
        for y_window in range(grid_size[0]):
            window_y = window_start + window_size * y_window - np.int(window_size/2)
            non_zero = np.nonzero(channel_image[window_y:window_y+window_size,:])
            if len(non_zero[1]) == 0:
                x = grid_size[1]
            else:
                mean_x = np.int(np.mean(non_zero[1]))
                x = mean_x/image.shape[1]*grid_size[1]
            grid_lane_x[y_window] = int(x)
        grid_data.append(foo(list(grid_lane_x)))
    return grid_data

def linear_points_into_grid_space(points, grid_size = (10, 36)):
    return None

def linear_interpolation_grid(data, no_lane_val):
    for lane_index in range(len(data)):
        lane_data = data[lane_index]
        for i in range(len(lane_data)-2):
            y = i + 1
            if lane_data[y] == no_lane_val:
                x_before = lane_data[y-1]
                if x_before == no_lane_val and y-2 >= 0:
                    x_before = lane_data[y-2]
                x_after = lane_data[y+1]
                if x_after == no_lane_val and y+2 < len(lane_data):
                    x_after = lane_data[y+2]
                if x_before != no_lane_val and x_after != no_lane_val:
                    y_coords = [y-1,y+1]
                    x_coords = [x_before, x_after]
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                    if m != 0.0:
                        inter_x = np.round((y/m)-(c/m))
                        data[lane_index][y] = int(inter_x)
    return data
