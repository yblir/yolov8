import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp20/weights/best.pt') # select your model.pt path
    model.predict(source='替换你的数据集地址',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )