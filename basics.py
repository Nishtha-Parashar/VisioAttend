#Basic steps which we follow for the actual program.

import cv2
import numpy as np
import face_recognition

imgkendall = face_recognition.load_image_file('Images_basics/kendall.jpeg')
imgkendall = cv2.cvtColor(imgkendall, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file('Images_basics/kendall_test.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgkendall)[0]
enKendall = face_recognition.face_encodings(imgkendall)[0]
cv2.rectangle(imgkendall,(faceloc[3],faceloc[0]), (faceloc[1],faceloc[2]), (255,0,255),2)

faceloctest = face_recognition.face_locations(img_test)[0]
enKendalltest = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test,(faceloctest[3],faceloctest[0]), (faceloctest[1],faceloctest[2]), (255,0,255),2)

result = face_recognition.compare_faces([enKendall],enKendalltest)
dist = face_recognition.face_distance([enKendall],enKendalltest)
print(result,dist)

cv2.imshow('Kendall Jenner', imgkendall)
cv2.imshow('Kendall Jenner test', img_test)
cv2.waitKey(0)