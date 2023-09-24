import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Celebs'
images =[]
name =[]
name_list = os.listdir(path)

#Loop for automatically getting all the images from the path folder into images list and all the names in the name_list.
for i in name_list:
    current = cv2.imread(f'{path}/{i}')
    images.append(current)
    name.append(os.path.splitext(i)[0])
print(name)

#Function to get all the encodings of each image and store it in the encodedList.
def findEncoding(images):
    encodedList = []
    for j in images:
        j = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(j)[0]
        encodedList.append(encode)
    return encodedList

# Function for Attendance part
def markAttendance(names):
    with open('Attendance.csv','r+') as f:
        myData = f.readlines()
        Name_List =[]
        for line in myData:
            entry = line.split(',')
            Name_List.append(entry[0])
        if names not in Name_List:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{names},{dtString}')


#This will tell us when all the encoding is done because it takes some time.
encodeList = findEncoding(images)
print('Encoding Complete')


#This will capture our live image with camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #Because we are doing it in real time, Image size reduction will help in speeding up the process.
    small = cv2.resize(img,(0,0),None, 0.25,0.25)
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    CurrentFrame = face_recognition.face_locations(small)
    CurFrameEncode = face_recognition.face_encodings(small,CurrentFrame)


#This loop will compare each encoding ans face location of the all the faces in the camera with the lists we already have.
    for encodeFace, faceLoc in zip(CurFrameEncode, CurrentFrame):
        match = face_recognition.compare_faces(encodeList, encodeFace)
        facedist = face_recognition.face_distance(encodeList, encodeFace)
        # print(facedist)
        matchIndex = np.argmin(facedist)

        if match[matchIndex]:
            name = name_list[matchIndex].upper()
            # print(names)
            y1,x2,y2,x1= faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)                       #Drawing rectangle around the face.
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,255,255),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_ITALIC,1,(255,0,0),2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)