

import time             
import cv2 as cv
import torch
import pandas as pd
    
model = torch.hub.load('./yolov5', 'custom', path='last.pt',force_reload=True, source='local')    # 커스텀한 yolov5s 모델 로드

'''
while True: 
    cap = cv.VideoCapture(0)    # 웹캠 on, 
    ret, frame = cap.read()      # cap.read()는 두 개의 반환값이 있다.  retval : 성공하면 True, 실패하면 False, frame: 현재 프레임 (numpy.ndarray)
    if not ret : continue        # cap,read() 실패 시, 다시 시도.
    break
'''

frame = cv.imread('aa.jpg')
results = model(frame)

val_df = results.pandas().xyxy[0]
val_df = val_df[val_df['confidence'] > 0.01]
val_list = val_df['name'].tolist()
    
print(results.pandas().xyxy[0])
print(val_list)     


# bounding box 그리기
for box in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = box
    label = f'{model.names[int(cls)]} {conf:.2f}'
    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv.putText(frame, label, (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# bounding box가 그려진 이미지 저장
cv.imwrite('aaa.jpg', frame)

    
