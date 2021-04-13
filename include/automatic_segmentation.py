#!/usr/bin/env python3

import numpy as np
import cv2

ROI_X_PADDING=20
MAX_POINTS=10

def extend_first_row(image, point):
    histogram = np.sum(image[image.shape[0]-50:,:], axis=0)
    if point[0] > 520: # if y of points is above threshold
        point_hist = histogram[point[1]-10: point[1]+10]
        if len(point_hist) > 0:
            point_x = np.argmax(point_hist) + point[1]+10
            if point_x > 0:
                return [image.shape[0]-1, point_x]
    return None

def automatic_segmentation(image, pts):
#    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_seg = np.zeros((image.shape[0], image.shape[1], 3))
    debug = image
    for index_lane in range(3):
        lane = pts[index_lane]
        if len(lane) > 1:
            first_point = extend_first_row(image, lane[0])
            if first_point != None:
                lane.insert(0,first_point)
        lane_color = [0,0,0]
        lane_color[index_lane] = 255
        number_points = len(lane)-1
        if number_points > MAX_POINTS:
            number_points = MAX_POINTS
        for index in range(number_points):
            point = lane[index]
            nextPoint = lane[index+1]
            lower_left = [point[1]-ROI_X_PADDING, point[0]]
            lower_right = [point[1]+ROI_X_PADDING, point[0]]
            upper_left = [nextPoint[1]-ROI_X_PADDING, nextPoint[0]]
            upper_right = [nextPoint[1]+ROI_X_PADDING, nextPoint[0]]
            roi = [upper_left, upper_right, lower_right, lower_left]
            cv2.fillPoly(mask_seg, [np.array(roi)], lane_color)
    segmented = cv2.bitwise_and(mask_seg, mask_seg, mask=image)
    return (segmented, debug)
