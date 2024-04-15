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
    results = model.predict(source=img,
                            imgsz=640,
                            # project='runs/detect',
                            # name='exp',
                            # classes=[0, 5, 11],
                            # save=True,
                            # show=True,
                            conf=0.25,
                            iou=0.5,
                            # conf_thres=0.5
                            # save_txt=True,
                            # show_boxes=True,
                            # show_conf=True,
                            # show_labels=True
                            )
    # cv2.waitKey()
    classes = results[0].boxes.cls.cpu().numpy()
    confidence = results[0].boxes.conf.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    print(classes)
    print(confidence)
    print(boxes)
    # print("----")
    # print(boxes[0][None])
    # print(boxes[classes == 0])
    print("=" * 50)
    a = bbox_ioa(boxes[0][None], boxes[classes == 0])
    # print(a[0])
    head_iou_max_bool, head_iou_index = np.max(a[0]) > 0.7, np.argmax(a[0])
    print(head_iou_max_bool, head_iou_index)
    if head_iou_max_bool:
        print(classes[head_iou_index], confidence[head_iou_index], boxes[head_iou_index])
    # print(detect_cls)
    # print(detect_boxes)
    # print(detect_conf)
    #
    # total_res = []
    # for i, cls in enumerate(detect_cls):
    #     # cur_det, cur_iou, cur_conf = [], [], []
    #     if cls == 5:
    #         x1, y1, x2, y2 = [int(item) for item in detect_boxes[i].cpu().numpy()]
    #         new_img = img[y1:y2, x1:x2]
    #         results2 = model.predict(source=new_img,
    #                                  imgsz=640,
    #                                  project='runs/detect2',
    #                                  save=True,
    #                                  name='exp',
    #                                  classes=[0, 11],
    #                                  )
    #         detect_cls2 = results2[0].boxes.cls
    #         detect_boxes2 = results2[0].boxes.xyxy
    #         detect_conf2 = results2[0].boxes.conf
    #         conf_list = [float(i) for i in detect_conf2.cpu().numpy()]
    #         max_conf = max(conf_list)
    #         max_index = conf_list.index(max_conf)
    #         object_cls = detect_cls2[max_index].cpu().numpy()
    #         object_box = detect_boxes2[max_index].cpu().numpy()
    #
    #         x11, y11, x12, y12 = object_box
    #         x11, y11, x12, y12 = x11 + x1, y11 + y1, x12 + x1, y12 + y1
    #         print(object_box)
    #         print(x11, y11, x12, y12)
