## AI야 교통안전을 부탁해 - 2021 인공지능 학습용 데이터 해커톤 대회
![KakaoTalk_20220709_230716837_01](https://user-images.githubusercontent.com/91238910/179466526-019ac225-6f71-4fb9-8b4b-6b9882595156.jpg)

 **제 목 :** 개인형이동장치(PM) 법규위반 영상 데이터를 활용한 이상 탐지/분류(Anomaly Detection/Classification) 프로그램 개발
 
 **기 간 :** 2021.11.18 ~ 2021.12.17
 
 **팀 명 :** 피하지말고 PM
 
 **팀 원 :** 박호준, 이주천, 장효주
  
 ![newslist_12](https://user-images.githubusercontent.com/8746262/147805256-d5ef10ae-59ba-46a4-9476-d0317dd37c32.jpg)
 
 <img width="50%" src="https://user-images.githubusercontent.com/8746262/147805946-6a79b21a-e44e-4af2-84bc-cf3d37174318.jpg"/>

 
 **수 상 :** 한국 폴리텍대학 이사장상 

 **기 사 :** https://news.naver.com/main/read.naver?mode=LSD&mid=sec&sid1=102&oid=215&aid=0001004291
 
 
## Summary

개인형 이동수단(PM)인 자전거, 전동킥보드, 이륜차의 교통법규 위반 탐지 프로그램입니다.
detecting와 tracking은 YOLOv5로, 위반 탐지 판단은 rule-based 알고리즘을 사용하였습니다.

### Presentation

 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893283-6655ccb3-848b-4fb3-ab7b-cbc07459ed0b.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893319-08a00775-06af-4d6e-bc93-2aa4a01f96fa.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893321-7e611fc8-1299-44a9-8f6b-9d7d0e1528ae.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893325-3a5f4cb6-44ec-417e-a27c-d365fd1bad40.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893333-6ae19323-20e1-41b4-88fc-96d1875b32e1.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893334-6dd5d221-1b7e-4ff2-8df1-84c17c79f5a9.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893338-b183883e-3911-4518-bf8b-e3b0dabd413a.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893340-771b97c4-f454-44e5-a908-6f5ec5aee44e.jpg"/>
 <img width="80%" src="https://user-images.githubusercontent.com/83095823/147893341-63f9fed5-4f09-4515-9fed-69fd7ad14fdf.jpg"/>

### code
##### 전처리 과정 1. yolo 학습을 위한 bounding box 좌표 변환
-> COCO 좌표 이므로, yolo 형식에 맞게 변환해줘야함
```python
def bbox(anno_info):
    bbox = []

    for m_bbox,cate_id in zip(anno_info['bbox'],anno_info['category_id']):
        xyxy=m_bbox

        b_center_x = (xyxy[0] + xyxy[2]) / 2
        b_center_y = (xyxy[1] + xyxy[3]) / 2
        b_width    = xyxy[2] - xyxy[0]
        b_height   = xyxy[3] - xyxy[1]

        b_center_x/=1920
        b_center_y/=1080
        b_width/=1920
        b_height/=1080
        cateid = cate_id -1 # 1-8 로 되어있어서 -1 해줘야함 (yolo는 class 를 0부터 읽음)

        bbox.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cateid, b_center_x, b_center_y, b_width, b_height))

    return bbox
```

##### 전처리 과정 2. 프레임 낮추기
구현 환경에 따라 과부하가 올 수 있기 때문에 프레임을 맞춰서 용량 문제 해결
```python
import cv2
import time
import imutils
from glob import glob
from PIL import Image
import os

# output 경로 지정
output_path = 'E:/train_person/images/' 
# 불러올 이미지 파일 경로 지정
load_path = 'C:/Users/82103/OneDrive/바탕 화면/교통문제 해결을 위한 CCTV 교통 영상(시내도로)/Training/교통안전(Bbox)/부천남부역 공영주차장/BC2000103/'
# 위에서 뽑아낸 label 파일 경로
label_path = 'E:/train_person/labels/'
# 파일명 리스트 불러와서
file = os.listdir(label_path)

# 이름만 가져와서 jpg 파일 붙인 후, load_path 에서 불러오기
file_list = [x.split(".")[0] + ".jpg" for x in file]

def save_img(file_list):
    
    for i in range(len(file_list)):
        cap = cv2.VideoCapture(load_path + file_list[i])
        hasFrame, img = cap.read()
        img_frame = imutils.resize(img, width=640, height = 360)
        cap.release()
        cv2.imwrite(output_path + file_list[i], img_frame)
        print(f'{i} th done!')
```

##### yolov5 적용하기
```python
from glob import glob
from sklearn.model_selection import train_test_split
import yaml
import os
import pandas as pd
import tensorflow as tf
import torch

train_img_list = glob('E:/new_file/images/train/*.jpg')
val_img_list = glob('E:/new_file/images/valid/*.jpg')

import os
os.chdir('C:/Users/82103/' + "yolov5")
import yolov5


# 이미지 이름에 맞는 텍스트 데이터 설정해주기
with open('E:/new_file/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')
with open('E:/new_file/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')
    
# yolo 학습을 위한 설명서 느낌의 yaml 파일 생성
data = {}
data['train'] = 'E:/new_file/train.txt'
data['val'] = 'E:/new_file/val.txt'

# model hyperparameter 지정
data['box'] = 0.05  # box loss gain 
data['cls_pw']= 1.0  # cls BCELoss positive_weight 
data['obj']= 1.0  # obj loss gain (scale with pixels) 
data['obj_pw'] = 1.0  # obj BCELoss positive_weight 
data['iou_t'] = 0.30  # IoU training threshold 
    
data['hsv_h']= 0.8  # image HSV-Hue augmentation (fraction) 
data['hsv_s']= 0.8  # image HSV-Saturation augmentation (fraction) 
data['hsv_v']= 0.8  # image HSV-Value augmentation (fraction) 
data['degrees']= 0.1  # image rotation (+/- deg) 
data['translate'] = 0.1  # image translation (+/- fraction) 
data['scale'] = 0.6  # image scale (+/- gain) 
data['shear'] = 0.01 # image shear (+/- deg) 

data['nc']=17

data['names']=['p_sign_green','p_sign_red','v_sign_green','v_sign_yellow',
'v_sign_red','v_sign_sl','v_sign_l','motor','byc','kickb','motor_r','byc_r','kickb_r','byc_c','kickb_c','kickb_many','pedestrian']

with open('C:/Users/82103/yolov5/data.yaml', 'w') as f:
    yaml.dump(data,f)

# pytorch gpu 사용법
import torch
torch.cuda.is_available()

from yolov5 import utils
display = utils.notebook_init()  # checks

# 모델 
!python train.py --img 640 --batch 8 --epochs 6 --data C:/Users/82103/yolov5/data.yaml --cfg C:/Users/82103/yolov5/models/yolov5s.yaml --weights yolov5s.pt --name test1212_yolov5s_object_results --device 0

# 학습된 모델로 탐지 
!python detect.py --weights E:/best_weights/object_best.pt --img 640 --conf 0.4 --source E:/도로교통공단\대용량데이터/PM사고위험영상데이터/횡단보도주행위반_5/자전거/정상/
```
