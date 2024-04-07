# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 14:03
# @Author  : yblir
# @File    : simple_predict.py
# explain  : 适用于部署的最简推理代码
# ======================================================================================================================
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.engine.exporter import Exporter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path_):
    model = attempt_load_one_weight(model_path_, device=device)


def load_model_engine(model_path_):
    exporter = Exporter()
    model = attempt_load_one_weight(model_path_, device=device)
    model.export(format="engine")
    # exporter(model,format="engine")

if __name__ == '__main__':
    model_path = "./weights/yolov8s.pt"
    load_model_engine(model_path_=model_path)
