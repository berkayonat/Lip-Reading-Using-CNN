import numpy as np
import cv2
from imutils import face_utils
import dlib 
import os


speakers = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08'] 
word_folder = ['01','02','03','04','05','06','07','08','09','10']
varieties = ['01','02','03','04','05','06','07','08', '09', '10']


os.mkdir('mouthNew')
for speaker in speakers:
   os.mkdir('mouthNew/'+speaker)
   
   for ind, folder in enumerate(word_folder):
     os.mkdir('mouthNew/'+speaker+'/' +folder)
     
     for vari in varieties:
       os.mkdir('mouthNew/'+speaker+'/' +folder+'/' +vari)
       image_list=os.listdir('D:/SeniorProject/dataset/'+speaker+'/'+'words/'+folder+'/'+vari)
       
       for im in image_list:
         image = cv2.imread('D:/SeniorProject/dataset/'+speaker+'/'+'words/'+folder+'/'+vari+'/'+im,0)
         face=dlib.get_frontal_face_detector()(image,1)
         
         for each in face:
           face_points=dlib.shape_predictor('D:/SeniorProject/shape_predictor_68_face_landmarks.dat')(image,each)
           face_points = face_utils.shape_to_np(face_points)
           (x, y, w, h) = cv2.boundingRect(np.array([face_points[49:68]]))     # 49-68 mouth points
           mouth = image[y-5:y+h+5, x-5:x+w+5]
           mouth = cv2.resize(mouth, (200, 100))
           print(speaker)
           print(folder+'-'+vari)
           #cv2_imshow(mouth)
           cv2.imwrite('mouthNew/'+speaker+'/' +folder+'/' +vari +'/' + im, mouth)