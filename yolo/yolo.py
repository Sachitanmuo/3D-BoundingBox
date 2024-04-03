"""
Will use opencv's built in darknet api to do 2D object detection which will
then get passed into the torch net

source: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
class cv_Yolo:

    def __init__(self, yolo_path, confidence=0.5, iou_threshold=0.3):
        self.model = YOLO(yolo_path)
        self.model.conf = confidence  
        self.model.iou = iou_threshold

        

    def detect(self, image_path):
        results = self.model(image_path)

        results = results[0]
        boxes = results.boxes.xyxy
        class_ids = results.boxes.cls
        detections = []
        confidences = []
        name = results.names

        for box, cls_id in zip(boxes, class_ids):
            #box = box[1]
            x1, y1, x2, y2, = box[0:4]
            box_2d = [(int(x1), int(y1)), (int(x2), int(y2))]
            class_ = name[cls_id.item()]
            if class_ == 'Pedestrian': 
                class_ = 'pedestrian'
            if class_ == 'Car': 
                class_ = 'car'
            print(class_)
            detections.append(Detection(box_2d, class_))
            '''
        for cls in enumerate(results.boxes.cls):
            class_ = name[cls[1].item()]
            print(class_)
            print(cls[1].item())
            '''
        return detections





class Detection:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_
    def get_2dbox(self):
        return self.box_2d
