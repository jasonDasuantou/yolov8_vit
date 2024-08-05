import os
from ultralytics import YOLO
import subprocess
from ultralytics.nn.vit.Vit import MbViTV3
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def add_vit(model):
    for name, param in model.model.named_parameters():
        stand = name[6:8]
        vit_ls = ['16']
        if stand in vit_ls:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(name)
    return model

def main():
    # model = YOLO(r'ultralytics/cfg/models/v8/yolov8x.yaml').load('/root/autodl-tmp/yolov8x.pt')
    model = YOLO(r'yolov8x_vit.yaml').load('runs/detect/vit/weights/vit.pt')
    model = add_vit(model)
    model.train(data="data.yaml", imgsz=640, epochs=50, batch=10, device=0, workers=0)
if __name__ == '__main__':
    main()