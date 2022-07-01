# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:54:14 2022

@author: Onat
"""

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from lipReading import Ui_MainWindow
from concat import concatFrames
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import shutil
import os
import sys

from tensorflow.keras.models import load_model


class lipReading(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.isSpeak=False
        self.control=True
        self.shapePredictorPath = 'shape_predictor_68_face_landmarks.dat'
        self.faceDetector = dlib.get_frontal_face_detector()
        self.facialLandmarkPredictor = dlib.shape_predictor(self.shapePredictorPath)
        self.ui.pushButton.clicked.connect(self.start_video)
        self.ui.btn_Open.clicked.connect(self.open_file)
        self.ui.pushButton_2.clicked.connect(self.predict)
        
    def start_video(self):
        self.isSpeak=True
        self.control=True
        for filename in os.listdir("frames"):
            filepath = os.path.join("frames", filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)

        self.cap = cv2.VideoCapture(0)
        #time.sleep(1.0)
        frame_count = 0
        TIMER = int(3)
        
        while self.control:
            prev = time.time()
 
            while TIMER >= 0:
                ret, frame = self.cap.read()
                frame = imutils.resize(frame, width=800)
                frame=cv2.flip(frame, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(TIMER),
                            (325, 325), font,
                            7, (0, 255, 255),
                            4, cv2.LINE_AA)
                cv2.imshow("WEBCAM", frame)
                cv2.waitKey(125)
                cur = time.time()
               
                if cur-prev >= 1:
                    prev = cur
                    TIMER = TIMER-1
            
            else:
                ret, frame = self.cap.read()
                frame = imutils.resize(frame, width=800)
                frame=cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
                cv2.namedWindow("WEBCAM")
                faces = self.faceDetector(gray, 0)
                for (i, face) in enumerate(faces):
                    facialLandmarks = self.facialLandmarkPredictor(gray, face)
                    facialLandmarks = face_utils.shape_to_np(facialLandmarks)
                    (x, y, w, h) = cv2.boundingRect(np.array([facialLandmarks[49:68]]))
                    mouth = gray[y-15:y+h+15, x-15:x+w+15]
                    mouth = cv2.resize(mouth, (200, 100))
                    mouth = mouth+5
    
                    if frame_count % 2 == 0 and frame_count < 53:
                        cv2.imwrite("frames/frame" + str(frame_count) + ".jpg", mouth)
                    frame_count += 1
                    
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    for (a, b) in facialLandmarks[49:68]:
                        cv2.circle(frame, (a, b), 1, (0, 0, 255), -1)
                
                
                cv2.imshow("WEBCAM", frame)
                key = cv2.waitKey(25)
                if key == ord("q"):
                    break
        self.cap.release()
        cv2.destroyWindow("WEBCAM")
           
        if self.isSpeak:
            for filename in os.listdir("input_frame"):
                filepath = os.path.join("input_frame", filename)
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)
                    
            concatFrames(2,53)
            self.ui.label_2.setText("Record completed successfully. Now you can predict.")
        
    def predict(self):
            
        class_names = ["BEGIN","CHOOSE","CONNECTION","NAVIGATION","NEXT","PREVIOUS","START","STOP","HELLO","WEB",
                       "STOP NAVIGATION","EXCUSE ME","I AM SORRY","THANK YOU","GOOD BYE","I LOVE THIS GAME",
                       "NICE TO MEET YOU","YOU ARE WELCOME","HOW ARE YOU ?","HAVE A GOOD TIME"]
    
        model = load_model("NewModel(85).h5")
        img = cv2.imread("input_frame/image.jpg",0)
        img = cv2.resize(img, (224,224))
        img = np.asarray(img)
        img = (img - np.min(img))/(np.max(img)-np.min(img))
        img = np.nan_to_num(img)
        #img = np.expand_dims(img, axis=2)
        img = img.reshape(1,224,224,1)
        
        y_pred = model.predict(img)
        print(y_pred)
        y_pred_class = np.argmax(y_pred, axis=1)
        print(y_pred_class)
        if y_pred[0][y_pred_class[0]]<0.4:
            self.ui.label_2.setText("Can you say again please ?")
        else:
            self.ui.label_2.setText(class_names[y_pred_class[0]])
                
                
    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        
        if fname:
            self.ui.lbl_Open.setText("File: "+ fname[0])
            self.isSpeak=True
            self.control=True
            for filename in os.listdir("frames"):
                filepath = os.path.join("frames", filename)
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)

            self.cap = cv2.VideoCapture(fname[0])
            #time.sleep(1.0)
            frame_count = 0
            
            while self.control:
                
                ret, frame = self.cap.read()
                if ret == False:
                    break
                frame = imutils.resize(frame, width=400, height=100)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
                cv2.namedWindow("FILE")
                faces = self.faceDetector(gray, 0)
                for (i, face) in enumerate(faces):
                    facialLandmarks = self.facialLandmarkPredictor(gray, face)
                    facialLandmarks = face_utils.shape_to_np(facialLandmarks)
                    (x, y, w, h) = cv2.boundingRect(np.array([facialLandmarks[49:68]]))
                    mouth = gray[y-15:y+h+15, x-15:x+w+15]
                    mouth = cv2.resize(mouth, (200, 100))
                
                    if frame_count % 5 == 0 and frame_count < 131:
                        cv2.imwrite("frames/frame" + str(frame_count) + ".jpg", mouth)
                    frame_count += 1
                    
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    for (a, b) in facialLandmarks[49:68]:
                        cv2.circle(frame, (a, b), 1, (0, 0, 255), -1)
                
                
                cv2.imshow("FILE", frame)
                key = cv2.waitKey(25)
                if key == ord("q"):
                    break
            self.cap.release()
            cv2.destroyWindow("FILE")
               
            if self.isSpeak:
                for filename in os.listdir("input_frame"):
                    filepath = os.path.join("input_frame", filename)
                    try:
                        shutil.rmtree(filepath)
                    except OSError:
                        os.remove(filepath)
                        
                concatFrames(5,131)
                self.ui.label_2.setText("Record completed successfully. Now you can predict.")
        
        
        
app=QApplication(sys.argv)
window=lipReading()
window.show()
app.exec_()
