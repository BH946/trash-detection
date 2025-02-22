# Intro
**Trash Detection Project with yolo/realsense/jetson-nano**

**쓰레기 탐지를 위해서 Yolov5, Yolov8 을 개발 및 테스트 해보고 이후에 하드웨어와 최종 결합합니다.**

* **데이터 셋 참고**
  * **[COCO to YOLO Dataset](https://github.com/BH946/cocoTOyolov5_dataset)**
  * **[googleapis-open datas](https://storage.googleapis.com/openimages/web/index.html), [roboflow-oopen datas](https://universe.roboflow.com/)**
  * **[data annotation 하기 좋은 사이트](https://www.cvat.ai/)**
* **공식 홈페이지**
  * **[Yolov5 Github](https://github.com/ultralytics/yolov5), [Yolov5 Docs](https://docs.ultralytics.com/yolov5)**
  * **[Yolov8 Github](https://github.com/ultralytics/ultralytics), [Yolov8 Docs](https://github.com/ultralytics/ultralytics)**
* **Yolo 이해하기 좋은 문서 2개**
  * **[[Object Detection] Architecture - 1 or 2 stage detector 차이](https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4)
  * **[[AI/딥러닝] 객체 검출(Object Detection) 모델의 종류 R-CNN, YOLO, SSD](https://rubber-tree.tistory.com/119)

<br><br>

# Trash Detection

**쓰레기 분류를 위해서는 단순히 "분류" 라기 보다는 이미지 내에 여러 객체들을 "탐지" 하는것이 중요합니다.**

**CNN의 모델을 객체 분류에 활용하면, 해당 하나의 이미지에 수많은 윈도우(=사각형 상자)들로 분류가 되기 때문에 한계가 있습니다.**

**이러한 한계를 해결하는 방법중에 하나가 "윈도우 일부만을 활용" 하는 방법이며, 이 방법에는 ''두가지 방식'' 이 있습니다.**

* **첫 번째 방식 : 영역 제안(region Proposal)**
  * 객체를 포함할 가능성이 높은 영역을 여러가지 알고리즘으로 제안
  * 해당 방식은 정확도를 제공하지만 처리 속도가 아래 방식보다 느리다.
  * 종류 : Faster R-CNN, R_FCN, FPN_FRCN 등등

* **두 번째 방식 : 정해진 위치, 크기의 객체만 찾기**
  * 정해진 위치와 크기를 미리 제공해서 해당 윈도우를 사용하는 방법
  * 해당 방식은 정확도는 위 방식보다 낮지만 처리 속도가 빠르다.
  * 종류 : YOLO, SSD, RetinaNet 등등

<br>

**쓰레기 분류의 정확도가 낮은건 직접 사람이 분류하거나, 따로 해당 데이터들만 추가학습을 해서 충분히 모델을 발 전시킬 수가 있습니다. 그러나 실시간의 처리를 원하는 저희의 입장에서 속도 개선은 더욱 어렵다고 생각이 들었습니다.   
이에 따라서 "두 번째 방식" 을 채택하였고, 그중에서 유명한 `YOLO` 를 사용하게 되었습니다.**

<br><br>

## 1. YOLO(You Only Look Once) 모델 선택

**`YOLO` 는 단일 단계 방식의 객체 탐지 알고리즘이며, 이미지 내에 객체와 객체 위치를 한 번만 보고 예측하기 때문에 속도가 빠른편에 속합니다.**

**참고로 수많은 버전들이 존재하는데 하나의 기업이나 사람이 개발한 것이 아닌 서로 다른 사람들이 개발한 오픈소스이기 때문에 버전별로 공식문서들이 전부 다르거나 없을수도 있습니다.**

**`Yolov5` 가 비교적 제일 유명한 모델이고, 제일 최근에 출시된 모델이 `Yolov8` 이므로 이 두가지를 사용하게 됩니다.**

* 버전별로 자세한 성능 비교는 공식 깃허브에 들어가면 볼 수 있습니다.
* 참고로 `ultralytics` 기업이 개발한 Yolo 모델버전이 5와 8이 해당됩니다.

* `Yolov8` 의 경우 **Detection, Segmentation, Classification, Pose** 모델이 제공되므로 이중에서도 선택해야합니다.
  * **Detectioin이 제일 기본이 되는 Yolo 모델이며 이것을 사용하게 됩니다.**
  * **Segmentation**은 복잡한 쓰레기 더미를 분류할 목적이였으면 이게 더 알맞는 모델이라고 생각하지만, 저희는 쓰레기를 한개 정도씩 넣어서 분류하는 수준이 목적이기 때문에 선택하지 않았습니다.
  * **Classification**은 전체적인 이미지 그림을 그저 분류하는 목적이면 사용하겠지만 저희처럼 하나의 이미지에 여러 쓰레기를 탐지해야하는 입장에서는 조금 알맞지 않습니다.
    * 예로 Cloudy, Sun, Rain 처럼 날씨 이미지를 분류할 때는 알맞다고 볼 수 있습니다.
    * 또한, 해당 방법은 Detectioin과 다르게 객체 위치를 구해주지 않습니다.
  * **Pose**는 모션(포즈)에 좀 더 집중된 모델로써 전혀 알맞지 않아서 사용하지 않습니다.

<br>

**따라서 저희의 상황에서 테스트를 위해 선택한 모델은 `Yolov5s, Yolov5x, Yolov8m` 입니다. 물론, 시간이나면 더 다양한 모델들을 테스트해 볼 예정입니다.** 

<br><br>

## 2. DataSet(데이터 셋)

**학습의 퀄리티를 책임지는것 중 하나는 "학습 데이터" 라고 생각하고, 가장 중요하다고 생각합니다.**

**Yolo는 학습 데이터를 아래와 같이 디렉토리 구조를 형성해야하는 특징이 있고, 제일 중요한것은 `images, labels` 하위 파일들의 이름이 동일해야한다는 점입니다.**

```bash
trash
 ┣ test
 ┃ ┣ images
 ┃ ┃ ┣ 1.jpg
 ┃ ┃ ┣ ...
 ┃ ┃ ┣ 81.jpg
 ┃ ┗ labels
 ┃ ┃ ┣ 1.txt
 ┃ ┃ ┣ ...
 ┃ ┃ ┣ 81.txt
 ┣ train
 ┃ ┣ images
 ┃ ┗ labels
 ┣ valid
 ┃ ┣ images
 ┃ ┗ labels
 ┗ data.yaml
```

* 물론 `data.yaml` 파일 속에서 해당 데이터들 경로를 직접 정의하기 때문에 꼭 위와 같은 디렉토리를 형성해야 할 필요는 없지만 
* `images, labels` 폴더는 꼭 같은 계층에 두고 두 폴더 하위 계층에 데이터는 꼭 동일한 파일 이름으로 `jpg, txt` 각각을 형성해야 합니다.

<br>

**참고로 본인은 `Roboflow, 네이버 커넥트 재단` 에서 쓰레기 데이터를 수집해서 사용했습니다.**

<br><br>

## 3. Train(학습)

**학습에는 다양한 옵션들이 존재하며 공식 문서에 잘 나와있습니다.**

**본인은 `Colab` 에서 학습을 진행했으며, `CLI, Python` 형태 둘다 사용이 가능하기 때문에   **
**Yolov5는 CLI, Yolov8은 Python으로 진행했습니다.**

**아래에서 자세한 설명은 하지않고, 간단하게만 학습과정을 소개하겠습니다.**

<br>

### Yolov5

**구글 드라이브를 연동합니다.**

```python
# drive mount
from google.colab import drive
drive.mount('/content/drive')
```

<br>

**Yolov5를 clone 합니다.**

```bash
# yolov5 downlaod
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt
```

<br>

**`data.yaml` 을 만듭니다.**

```python
# 원하는 입맛대로 변경
import yaml
data = { 
    'train': ['trash_real/train/images'],
    'test': ['trash_real/test/images'],
    'val': ['trash_real/valid/images'],
    'names': ['Plastic', 'Metal'],
    'nc': 2
}
```

<br>

**원하는 옵션을 설정해서 `학습(train)` 을 진행합니다.**

* 학습이 종료되면 `runs/train` 경로에 여러가지 학습 결과를 볼 수 있으며, 모델도 이곳에 있습니다.
* 참고로 학습에는 train 이미지만 사용이 되며, 마무리 단계에서 valid 이미지를 사용하여 검증을 진행해줍니다.

```bash
!python train.py --img 340 --batch 40 --epochs 50 --data ./{INPUT_FOLDER_NAME}/data_test.yaml --weights yolov5x.pt --cache
```

<br>

**위에서 학습한 `best.pt` 모델로 `detect` 를 수행해봅니다.**

```bash
# 테스트 이미지 전체를 detect 수행 해보는 것
!python detect.py --weights runs/train/{TRAIN_FOLDER_NAME}/weights/best.pt --img 640 --conf 0.4 --source ./{INPUT_FOLDER_NAME}/test/images --name {DETECT_FOLDER_NAME}
```

<br><br>

### Yolov8

**구글 드라이브를 연동합니다.**

```python
# drive mount
from google.colab import drive
drive.mount('/content/drive')
```

**Yolov8를 clone 합니다.**

```python
# import torch
# torch.cuda.empty_cache()
!pip install ultralytics
```

<br>

**`data.yaml` 을 만듭니다.**

```python
# 테스트용 데이터
BASEPATH = '/content/drive/MyDrive/Project/4_1_Semester_Advancement_of_College_Education/yolov8'
import os
import yaml
data = {
    'path': f'{BASEPATH}',
    'train': [os.path.join('data_test', 'train', 'images')],
    'val': [os.path.join('data_test', 'valid', 'images')],
    'names': ['Plastic', 'Metal'],
}
with open(os.path.join(BASEPATH, 'config_test.yaml'),'w') as f:
  yaml.safe_dump(data, f)
with open(os.path.join(BASEPATH, 'config_test.yaml'),'r') as f:
  dataYaml = yaml.safe_load(f)
print(dataYaml)
```

<br>

**원하는 옵션을 설정해서 `학습(train)` 을 진행합니다.**

* 참고로 학습에는 train 이미지만 사용이 되며, 마무리 단계에서 valid 이미지를 사용하여 검증을 진행해줍니다.

```python
import os
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")

# Use the model
results = model.train(data=os.path.join(BASEPATH, 'config_test.yaml'), epochs=3, imgsz=340, name=os.path.join(BASEPATH,'runs_test', 'run'))
```

<br>

**위에서 학습한 `best.pt` 모델로 `detect` 를 수행해봅니다.**

```python
import cv2
import os
from ultralytics import YOLO

# Load a model
model_path = os.path.join('.', 'best.pt')
model = YOLO(model_path)

# Use the model
img_path = os.path.join('.', 'img_test.jpg')
results = model.predict(source=img_path, save=False, imgsz=340)[0]
res_plotted = results.plot()
cv2.imshow("result", res_plotted)
```

<br><br>

## 4. 하드웨어 통신

**`jetson-nano` 와 `arduino` 를 USB 연결 후 `Serial` 통신을 사용합니다.**

* `jetson-nano` 대신에 `노트북(컴퓨터)` 로도 대신 통신할 수 있습니다.
* `jetson-nano` 를 사용간에 주의할 점이 있습니다.
  * **젯슨나노**는 **SD카드**를 따로 삽입해서 메모리를 사용해야 합니다.
  * **젯슨나노**만 작동시킬 때는 10w를 사용하므로 5V-2A로 동작이 가능하지만 모니터, 마우스, 키보드 등 기타 주변장치를 연결할 경우 **배럴잭**을 이용한 전원 공급이 필요합니다.
    * **배럴잭**을 이용한 전원 공급 시 **J48핀을 쇼트**
    * **J48핀은 전원 모듈과 연결**되어 있어 쇼트 시 배럴잭으로 전원 공급이 가능
* `realsense` 카메라를 사용할 때도 주의할 점이 있습니다.
  * 반드시 **USB 포트 3.0** 을 사용해야 합니다.
    * 참고로 젯슨나노는 3.0포트를 지원합니다.
  * 해당 카메라는 일반 카메라와 깊이 카메라를 지원합니다.

<br>

**실제 구동한 코드(아두이노 코드는 생략)**

```python
# python : 3.9.16
import time
import os

from ultralytics import YOLO
import cv2

import serial # 라이브러리 제공
arduino = serial.Serial('com4', 9600) # 추가 - 아두이노

cap = cv2.VideoCapture(0) # 0번째 device id -> realsense 카메라 순번으로 설정
ret, frame = cap.read()
H, W, _ = frame.shape
model_path = os.path.join('.', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.8 # 임계값 설정 0.8 이하 확률은 무시하려고

class_name_dict = {0: 'plastic', 1: 'metal'}

detectCount = 0
while ret:
    # 파라미터 참고 : https://docs.ultralytics.com/modes/predict/#inference-arguments
    results = model.predict(source=frame, save=False, imgsz=340)[0]

    if len(results.boxes)==0: detectCount=0
    else:
        for result in results.boxes.data.tolist(): # 탐지때만 for문 들어가는 것
            x1, y1, x2, y2, score, class_id = result
            if score < threshold: detectCount=0
            else: # score >= threshold
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                detectCount+=1
            
            if detectCount >= 3: # 총 3번 확인(아래 sleep으로 속도 설정)
                # 하드웨어 동작
                print(class_id)
                detectCount=0
                str1 = str(int(class_id)+1)
                arduino.write(str1.encode()) # 추가 - 아두이노
                time.sleep(5) # 추가 - 아두이노

    cv2.imshow('camera', frame) # show -> 테스트 할 때 확인용
    if cv2.waitKey(1) == ord('q'): # 사용자가 q 를 입력하면 
        break
    
    time.sleep(0.5) # 임의로 속도조절 추가
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
```

<br><br>

## 5. 테스트

**실제 테스트 모습**

* Metal로 인식후 모터가 오른쪽으로 회전해서 쓰레기를 분류해줍니다.

<img src="https://github.com/BH946/trash-detection/assets/80165014/0396ea8f-b576-4735-be14-c0c3134f01a3" alt="test" style="zoom:80%;" />  

<br><br>

# Folder Structure

* [`/main_predict.py`](./main_predict.py)
  * Detect 동작을 진행하는 메인 함수
* [`/runs/detect/train/`](./runs/detect/train/)
  * 실제 메인 함수를 동작시켜서 얻은 결과이며, weights 폴더에 저장된 학습 모델은 용량 때문에 제외
