#!/usr/bin/env python3
# VERSION 2

import numpy as np
import cv2
def foo(l, dtype=int):
    return list(map(dtype, l))

def segmented_image_into_grid_space(image, grid_size = (10, 36), window_size_x=10):
    w = grid_size[1]+1
    window_size_y = int(image.shape[0]/grid_size[0])
    grid_data = []
    for channel_index in range(image.shape[2]):
        channel_image = image[:,:,channel_index]
        y_data = []
        for y_window in range(grid_size[0]):
            window_y = window_size_y * y_window
            x_row = np.zeros(w)
            for x_window in range(int(image.shape[1]/window_size_x)):
                    window_x = window_size_x * x_window
                    non_zero = np.nonzero(channel_image[window_y:window_y+window_size_y, window_x:window_x+window_size_x])
                    # if all values are zero, there is no line
                    if len(non_zero[1]) > 0:
                        mean_x = np.int(np.mean(non_zero[1])) + window_x
                        x = int(mean_x/image.shape[1]*grid_size[1])
                        x_row[x] = 1.0
            if len(np.nonzero(x_row)) == 0:
                x_row[w-1] = 1.0
            y_data.append(x_row)
        grid_data.append(y_data) 
       # grid_data.append(foo(list(grid_lane_x)))
    return np.array(grid_data)

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
