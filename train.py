import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8_modify/yolov8s_lyb.yaml')
    # model=YOLO(r"E:\GitHub\yolov8\ultralytics\cfg\models\v8_modify\yolov8s-ODConv-Gold-AFPN.yaml")
    model=YOLO(r'ultralytics/cfg/models/v8_modify/yolov8s_4fpn.yaml')
    # model = YOLO(r"E:\GitHub\yolov8\ultralytics\cfg\models\v8_modify\yolov8s.yaml")
    model.load('weights/yolov8s.pt')  # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

    model.train(data='ultralytics/cfg/datasets/coco128.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                task='detect',
                # scale='l',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=12,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )
