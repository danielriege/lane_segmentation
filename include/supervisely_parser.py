#!/usr/bin/env python3
# VERSION 2

import json
import cv2
import numpy as np

def getPoints(annotation_path):
    relative_path = annotation_path
    lanes = {}
    with open(relative_path) as f:
        data = json.load(f)
    i = 0
    for object_lane in data['objects']:
        lane_points = []
        for points in object_lane['points']['exterior']:
            lane_points.append([points[0], points[1]])
        class_name = object_lane['classTitle']
        class_name = f"{class_name}.{i}"
        i += 1
        lanes[class_name] = lane_points
    return lanes

def drawLanes(img, lanes):
    img = np.zeros_like(img)
    for class_title in lanes:
        if "outer" in class_title:
            color = (1.0,0,0)
        else:
            color = (0,1.0,0.0)
        pts = np.asarray(lanes[class_title], np.int32)
        img = cv2.polylines(img, [pts], False, color,1)
    return img
