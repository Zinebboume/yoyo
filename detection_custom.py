
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *


#image_path   = "C:/Users/boume/Downloads/deeplearning/TensorFlow-2.x-YOLOv3/IMAGES"
image_path='C:/Users/dsben/OneDrive/Source/Source/Comptage_De_Vache/Projet_Finale_Comptage_De_Vache/TensorFlow-2.x-YOLOv3/IMAGES/25.jpg'
image_path_output='C:/Users/dsben/OneDrive/Source/Source/Comptage_De_Vache/Projet_Finale_Comptage_De_Vache/TensorFlow-2.x-YOLOv3/IMAGES/25.jpg'
# Load the image using cv2.imread()
image = cv2.imread(image_path)

#image_path   = "C:/Users/boume/Downloads/deeplearning/TensorFlow-2.x-YOLOv3/IMAGES"
#video_path   = "C:/Users/boume/Downloads/deeplearning/TensorFlow-2.x-YOLOv3/IMAGES"

yolo = Load_Yolo_model()
detect_image(yolo, image, image_path_output,input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, "C:/Users/lbenidiri/IA/PYCH/TensorFlow-2.x-YOLOv3/IMAGES/Les vaches retournent au pâturage !.mp4", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, 'video_path', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
