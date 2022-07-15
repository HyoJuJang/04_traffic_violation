## AI야 교통안전을 부탁해 - 2021 인공지능 학습용 데이터 해커톤 대회


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
##### -> COCO 좌표 이므로, yolo 형식에 맞게 변환해줘야함
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

전처리 과정 2. 프레임 낮추기

```python
```
