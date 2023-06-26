# python : 3.9.16 (tf) 
import time
import os

from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0) # 0번째 device id
ret, frame = cap.read()
H, W, _ = frame.shape
model_path = os.path.join('.', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.8 # 임계값 설정 0.8 이하 확률은 무시하려고

class_name_dict = {0: 'plastic', 1: 'metal'}

# import serial
# arduino = serial.Serial('com4', 9600) # 추가 커스텀 - 아두이노

detectCouont = 0
while ret:
    # 파라미터 : https://docs.ultralytics.com/modes/predict/#inference-arguments
    results = model.predict(source=frame, save=False, imgsz=340)[0]

    if len(results.boxes)==0: detectCouont=0
    else:
        for result in results.boxes.data.tolist(): # 탐지때만 for문 들어가는 것
            x1, y1, x2, y2, score, class_id = result
            if score < threshold: detectCouont=0
            else: # score >= threshold
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                detectCouont+=1
            
            if detectCouont >= 3: # 총 3번 확인(아래 sleep으로 속도 설정)
                # 하드웨어 동작
                print(class_id)
                detectCouont=0
                # str1 = str(int(class_id)+1)
                # arduino.write(str1.encode()) # 추가 커스텀 - 아두이노
                # time.sleep(5) # 추가 커스텀 - 아두이노
    print(detectCouont)
    cv2.imshow('camera', frame) # show
    if cv2.waitKey(1) == ord('q'): # 사용자가 q 를 입력하면 
        break
    
    time.sleep(0.5) # 임의로 추가
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
