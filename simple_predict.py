# -*- coding: utf-8 -*-
# @File  : yolo_inference.py
# @Author: yblir
# @Time  : 2024-04-14 8:54
# @Explain:
# ======================================================================================================================
import os.path
import random
import time
import cv2
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
from typing import List, Tuple, Union
from numpy import ndarray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


class YOLOV8DetectionInfer:
    def __init__(self, weights, conf_thres, iou_thres) -> None:
        self.imgsz = 640
        self.model = AutoBackend(weights, device=device)
        self.model.eval()
        self.names = self.model.names
        self.half = False
        self.conf = conf_thres
        self.iou = iou_thres
        self.color = {"font": (255, 255, 255)}
        # self.color.update(
        #         {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #          for i in range(len(self.names))})
        colors_ = Colors()
        self.color.update(
                {self.names[i]: colors_(i) for i in range(len(self.names))}
        )

    def infer(self, img_path, save_path):
        img_src = cv2.imread(img_path)
        img = self.precess_image(img_src, self.imgsz, self.half, device)
        # t1 = time.time()
        preds = self.model(img)
        # t2 = time.time()
        det = ops.non_max_suppression(preds, self.conf, self.iou,
                                      classes=None, agnostic=False, max_det=300, nc=len(self.names))
        # t3 = time.time()
        return_res = []
        for i, pred in enumerate(det):
            # lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
            # tf = max(lw - 1, 1)  # font thickness
            # sf = lw / 3  # font scale
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
            results = pred.cpu().detach().numpy()
            for result in results:
                return_res.append([result[:4], result[4], int(result[5])])
                # self.draw_box(img_src, result[:4], result[4], self.names[result[5]], lw, sf, tf)

        # cv2.imwrite(os.path.join(save_path, os.path.split(img_path)[-1]), img_src)
        return return_res
        # return (t2 - t1) * 1000, (t3 - t2) * 1000

    def draw_box(self, img_src, box, conf, cls_name, lw, sf, tf):
        color = self.color[cls_name]
        label = f'{cls_name} {round(conf, 3)}'
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制矩形框
        cv2.rectangle(img_src, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        # text width, height
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        # label fits outside box
        outside = box[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 绘制矩形框填充
        cv2.rectangle(img_src, p1, p2, color, -1, cv2.LINE_AA)
        # 绘制标签
        cv2.putText(img_src, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)

    @staticmethod
    def letterbox(im: ndarray,
                  new_shape: Union[Tuple, List] = (640, 640),
                  color: Union[Tuple, List] = (114, 114, 114),
                  stride=32) -> Tuple[ndarray, float, Tuple[float, float]]:
        # todo 640x640,加灰度图
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # new_shape: [width, height]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        # Compute padding [width, height]
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

        # todo 这步操作,能填充一个包裹图片的最小矩形,相当于动态shape, 输出目标的置信度与较大偏差
        # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def precess_image(self, img_src, img_size, half, device):
        # Padded resize
        img = self.letterbox(img_src, img_size)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img


if __name__ == '__main__':
    # weights = r'yolov8n.onnx'
    weights = r'yolov8m.pt'
    save_path = "./runs"

    model = YOLOV8DetectionInfer(weights, 0.45, 0.45)

    img_path = r"./ultralytics/assets/bus.jpg"
    res = model.infer(img_path, save_path)

    for i in res:
        print(i)

    # infer_time, nms_time = 0, 0
    # for i in range(1000):
    #     res = model.infer(img_path, save_path)
    #     infer_time += res[0]
    #     nms_time += res[1]
    # print(round(infer_time / 1000, 2), round(nms_time / 1000, 2))

# pt
# 4.51 0.65
# 4.24 1.64
# 5.05 1.56
# 5.46 1.46
# 4.39 1.61
# 5.44 1.33
# 4.26 1.75
# 4.1 1.7

# onnx
# 13.97 1.33
# 8.68 1.19
# 8.91 1.15
# 11.66 1.25
# 13.66 1.32
# 11.99 1.32
