#!/usr/bin/env python3

import json
import cv2
import numpy as np

IMAGE_H = 480
IMAGE_W = 848
yDecompressionFactor = 250
src = np.float32([[0, 0], [IMAGE_W, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
dst = np.float32([[0, 0], [IMAGE_W, 0], [360, IMAGE_H + yDecompressionFactor], [480, IMAGE_H + yDecompressionFactor]])
matrix = cv2.getPerspectiveTransform(src, dst) # The transformation matrix

def getPoints(input_path, image_filename):
    relative_path = '{}/ann/{}.json'.format(input_path, image_filename)
    lanes = [[], [], []]
    with open(relative_path) as f:
        data = json.load(f)
    for object_lane in data['objects']:
        lane_points = []
        for points in object_lane['points']['exterior']:
#            lane_points.append([points[0], points[1]])
#            continue
            point = []
            y = points[1] - 60
            x = points[0]
            px = (matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]) / ((matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))
            py = (matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]) / ((matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]))

            point.append(int(px))
            point.append(int(py))
            lane_points.append(point)
        if object_lane['classTitle'] == 'left':
            lanes[0] = lane_points
        elif object_lane['classTitle'] == 'middle':
            lanes[1] = lane_points
        elif object_lane['classTitle'] == 'right':
            lanes[2] = lane_points
    return lanes

def drawLanes(img, points):
    img = np.zeros_like(img)
    left, middle, right = points

    pts_l = np.asarray(left, np.int32)
    img = cv2.polylines(img, [pts_l], False, (255,0,0),6)

    pts_m =  np.asarray(middle, np.int32)
    img = cv2.polylines(img, [pts_m], False, (0,255,0),6)

    pts_r = np.asarray(right, np.int32)
    img = cv2.polylines(img, [pts_r], False, (0,0,255),6)
    return img
