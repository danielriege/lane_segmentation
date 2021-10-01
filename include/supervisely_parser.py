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

def drawDebugImage(img, lanes):
    img = np.zeros_like(img)
    for class_title in lanes:
        if "outer" in class_title:
            color = (1.0,0,0)
        else:
            color = (0,1.0,0.0)
        pts = np.asarray(lanes[class_title], np.int32)
        img = cv2.polylines(img, [pts], False, color,5)
    return img

def drawLanes(size, lanes):
    data = [np.zeros(size, dtype=np.int32) for i in range(6)]
    for class_title in lanes:
        if "outer_l" in class_title:
            i = 0
        elif "outer_t" in class_title:
            i = 0
        elif "outer_r" in class_title:
            i = 0
        elif "middle_curb" in class_title:
            i = 1
        elif "guide_lane" in class_title:
            i = 2
        elif "solid_lane" in class_title:
            i = 3
        elif "hold_line" in class_title:
            i = 4
        elif "zebra" in class_title:
            i = 5
        else:
            return None
        pts = np.asarray(lanes[class_title], np.int32)
        if "area" in class_title or "zebra" in class_title:
            data[i] = cv2.fillPoly(data[i], [pts], 1)
        else:
            thick = 1
            if "hold" in class_title:
                thick = 3
            data[i] = cv2.polylines(data[i], [pts], False, 1,thick)
    # make sure we are only occuping free pixels
    for index in range(1,len(data)-1):
        data[index] = np.bitwise_and(np.invert(data[index-1]), data[index])
    # background = cv2.absdiff(np.ones(size, dtype=np.int32), data[0])
    # background = cv2.absdiff(background, data[1])
    # background = cv2.absdiff(background, data[2])
    # background = cv2.absdiff(background, data[3])
    # background = cv2.absdiff(background, data[4])
    # background = cv2.absdiff(background, data[5])
    # data.append(background)

    
    return data
