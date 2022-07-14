import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not cap.isOpened():
        cap=cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Cannot Open webcam")
    result=DeepFace.analyze(frame,actions=['age'])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces=faceCascade.detectMultiScale(gray,1.2,4)
    # for (x,y,w,h) in faces:
    #     print(x,y,w,h)
        # roi_gray=gray[y:y+h,x:x+w]
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
    font=cv2.FONT_HERSHEY_SIMPLEX#now we insert text into video
    cv2.putText(frame,str(result['age']),
         (0,50),font,3,(0,0,255),4,cv2.LINE_4)
    cv2.imshow('Demo video',frame)
    if cv2.waitKey(20)& 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
