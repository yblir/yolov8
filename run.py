# -*- coding: utf-8 -*-
# @File  : run.py
# @Author: yblir
# @Time  : 2024-03-23 9:37
# @Explain: 
# ======================================================================================================================
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model

    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("ultralytics/assets/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format

    res = model.predict("ultralytics/assets/bus.jpg", save=True)
    print(res)
