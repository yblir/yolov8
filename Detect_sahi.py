import warnings

import cv2
import numpy as np

warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_ioa

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

if __name__ == '__main__':
    model = YOLO('weights/yolov8n.pt')  # select your model.pt path
    # a=model.export(format='engine')
    img = cv2.imread("ultralytics/assets/bus.jpg")
    detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path="weights/yolov8n.pt", confidence_threshold=0.5, device="gpu"
    )
    results = get_sliced_prediction(
            img, detection_model, slice_height=256, slice_width=256, overlap_height_ratio=0.2, overlap_width_ratio=0.2
    )
    object_prediction_list = results.object_prediction_list

    boxes_list = []
    clss_list = []
    conf_list = []
    for ind, _ in enumerate(object_prediction_list):
        boxes = (
            object_prediction_list[ind].bbox.minx,
            object_prediction_list[ind].bbox.miny,
            object_prediction_list[ind].bbox.maxx,
            object_prediction_list[ind].bbox.maxy,
        )
        clss = object_prediction_list[ind].category.id
        conf = object_prediction_list[ind].score.value

        boxes_list.append(boxes)
        clss_list.append(clss)
        conf_list.append(conf)
    a=np.array(boxes_list)
    print(a)
    print(type(a[0]))
    print(np.array(clss_list))
    print(np.array(conf_list))

# [          5           0           0           0           0          11]
# [    0.87345     0.86569     0.85284     0.82522     0.26111     0.25507]
# [[     22.871      231.28         805      756.84]
#  [      48.55      398.55      245.35       902.7]
#  [     669.47      392.19      809.72      877.04]
#  [     221.52       405.8      344.97      857.54]
#  [          0      550.53      63.007      873.44]
#  [   0.058174      254.46      32.557      324.87]]
# ==================================================
# True 2
# 0.0 0.85283536 [     669.47      392.19      809.72      877.04]
