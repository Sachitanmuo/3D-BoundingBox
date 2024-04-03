import torch
from yolo.yolo import cv_Yolo 
import os
yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
model = cv_Yolo (yolo_path)
model.load_state_dict(torch.load('yolov8n.pt'))
model.eval()


dummy_input = torch.randn(1, 3, 416, 416)  

onnx_path = 'yolo_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])

