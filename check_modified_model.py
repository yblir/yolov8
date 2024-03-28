# -*- coding: utf-8 -*-
# @File  : check_modified_model.py
# @Author: yblir
# @Time  : 2024-03-24 11:04
# @Explain: 
# ======================================================================================================================
# Ultralytics YOLO 🚀, AGPL-3.0 license

# import contextlib
# from copy import copy
# from pathlib import Path
#
# import cv2
# import numpy as np
# import pytest
# import torch
# import yaml
# from PIL import Image
# from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
# from ultralytics.cfg import TASK2DATA
# from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,

    WEIGHTS_DIR,

)

# MODEL = WEIGHTS_DIR / "path with spaces" / "yolov8n.pt"  # test spaces in path
# CFG = "yolov8n.yaml"


SOURCE = ASSETS / "bus.jpg"


# TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
# IS_TMP_WRITEABLE = is_dir_writeable(TMP)


def model_forward(CFG):
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG, verbose=True)
    model(source=SOURCE)  # also test no source and augment


if __name__ == '__main__':
    # 检测修改配置文件yaml后生成的的model是否能正常运行
    # yaml_path = 'ultralytics/cfg/models/v8_modify/yolov8-AKConv.yaml'
    yaml_path= 'ultralytics/cfg/models/v8_modify/yolov8s_lyb.yaml'
    model_forward(yaml_path)
