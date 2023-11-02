from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import argparse

import tensorflow as tf

import io
import time
import datetime
import csv
import requests

import numpy as np

import tkinter
import tkinter.font as tkFont

from PIL import Image
import PIL.Image, PIL.ImageTk

import pygame

from imutils.video import VideoStream
import imutils
import pickle
import os

class App:
    def __init__(self, window, window_title, video_source_0=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('500x570')
        self.window.resizable(False, False)

	# ===== 以下這個區塊主要是設定臉部辨識的所需匯入的檔案 ===== #
        self.protoPath = 'face_detection_model/deploy.prototxt'
        self.modelPath = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        self.embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
        self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
        self.le = pickle.loads(open("output/le.pickle", "rb").read())
	# ===== 以上這個區塊主要是設定臉部辨識的所需匯入的檔案 ===== #

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--model', help='File path of .tflite file.', default = 'model.tflite')
        parser.add_argument(
            '--labels', help='File path of labels file.', default = 'labels.txt')
        args = parser.parse_args()

        self.labels = self.load_labels(args.labels)
        self.interpreter = tf.lite.Interpreter(args.model)
        self.interpreter.allocate_tensors()
        self._, self.height, self.width, self._ = self.interpreter.get_input_details()[0]['shape']

        self.video_source_0 = video_source_0

        self.fontStyle = tkFont.Font(size=30)
        self.header_label = tkinter.Label(window, text='影像識別 GUI設計',font=self.fontStyle)
        self.header_label.place(x=1, y=5, width=500, height=40)

        # 建立日期與時間內容資訊標題
        self.fontStyle = tkFont.Font(size=12)
        self.date_label = tkinter.Label(window, text='日期', bg="blue", fg="yellow", font=self.fontStyle)
        self.date_label.place(x=60, y=50, width=189, height=24)

        self.fontStyle = tkFont.Font(size=12)
        self.time_label = tkinter.Label(window, text='時間', bg="blue", fg="yellow", font=self.fontStyle)
        self.time_label.place(x=254, y=50, width=189, height=24)

        # 建立攝影機使用
        self.vid_0 = MyVideoCapture(self.video_source_0)
  
        # 建立即時影像畫布
        self.canvas_face = tkinter.Canvas(window, width = 240, height = 180)
        self.canvas_face.place(x=20, y=130, width=240, height=180)

        # 設定臉部識別按鈕的樣式與位置
        self.fontStyle = tkFont.Font(size=16)
        self.face_button = tkinter.Button(window, text="臉部訓練", font=self.fontStyle, command=self.face_train)
        self.face_button.place(x=50, y=350, width=160, height=60)

        # 設定臉部訓練按鈕的樣式與位置
        self.fontStyle = tkFont.Font(size=16)
        self.face_button = tkinter.Button(window, text="臉部識別", font=self.fontStyle, command=self.add_face_name_list)
        self.face_button.place(x=50, y=420, width=160, height=60)
        
        # 設定影像識別按鈕的樣式與位置
        self.fontStyle = tkFont.Font(size=16)
        self.before_class_button = tkinter.Button(window, text="影像識別", font=self.fontStyle, command=self.image_class)
        self.before_class_button.place(x=50, y=490, width=160, height=60)

        # 建立識別狀況框架區
        self.attend_status_frame = tkinter.Frame(window)
        self.attend_status_frame.place(x=280, y=130, width=200, height=420)
        self.update_attend_status()

        self.update_clock()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.hint_flag = 0

        self.window.mainloop()

    def update_attend_status(self):
        # 建立出席狀況區框架裡的卷軸功能
        sb_status = tkinter.Scrollbar(self.attend_status_frame)
        sb_status.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        # 建立一個出席狀況清單
        fontStyle = tkFont.Font(size=16)
        # 將選項框在Y軸的動作與捲軸進行關聯
        self.attend_status_listbox = tkinter.Listbox(self.attend_status_frame, height=8, 
            yscrollcommand = sb_status.set, font=fontStyle)  
        self.attend_status_listbox.pack(side=tkinter.LEFT, fill=tkinter.BOTH)

    def face_train(self):
        os.system('python3 extract_embeddings.py\
            --dataset dataset\
            --embeddings output/embeddings.pickle\
            --detector face_detection_model\
            --embedding-model openface_nn4.small2.v1.t7')
        os.system('python3 train_model.py\
            --embeddings output/embeddings.pickle\
            --recognizer output/recognizer.pickle\
            --le output/le.pickle')
	# ===== 以下這個區塊主要是設定臉部辨識的所需匯入的檔案 ===== #
        self.protoPath = 'face_detection_model/deploy.prototxt'
        self.modelPath = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        self.embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
        self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
        self.le = pickle.loads(open("output/le.pickle", "rb").read())
	# ===== 以上這個區塊主要是設定臉部辨識的所需匯入的檔案 ===== #
            
    def face(self):
        # Get a frame from the video source
        ret_0, frame_0 = self.vid_0.get_frame()

        if (ret_0):
            self.Face_detection(frame_0)

    def add_face_name_list(self):
        self.name=''
        # attend_status_value = self.attend_value.get()
        self.face()
        if (self.name != ''):
            name_value = self.name
            self.attend_status_listbox.insert(tkinter.END, name_value + ' ' + 
                time.strftime("%H") + ':' + time.strftime("%M") + ':' + time.strftime("%S"))

    def image_class(self):
        # Get a frame from the video source
        ret_0, frame_0 = self.vid_0.get_frame()

        if (ret_0):
            self.classification(frame_0)

        self.hint_Window()

            
    def hint_Window(self):
        self.hint = tkinter.Toplevel(self.window)
        self.hint.geometry('400x200')
        self.hint.resizable(False, False)

        fontStyle = tkFont.Font(size=18)
        self.hint_info = tkinter.Text(self.hint,height=50,font=fontStyle, bg='yellow',fg='red')

        self.hint_flag = self.hint_flag + 1
        self.hint_text = 'No. ' + str(self.hint_flag)
        self.hint_info.insert('insert',self.hint_text)

        self.hint_text = '\n' + self.image_text
        self.hint_info.insert('insert',self.hint_text)

        self.hint_text = '\n' + 'Inference time= ' + str(round(self.inference_time*1000,2)) + 'ms'
        self.hint_info.insert('end',self.hint_text)

        self.hint_info['state'] =tkinter.DISABLED
        self.hint_info.pack()

    def update(self):
        # Get a frame from the video source
        ret_0, frame_0 = self.vid_0.get_frame()

        frame_0_small=cv2.resize(frame_0,(240,180))

        if ret_0:
            self.photo_0 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_0_small))
            self.canvas_face.create_image(1,0,image = self.photo_0, anchor="nw")

        self.window.after(self.delay, self.update)

    def update_clock(self):
        now = time.strftime("%H:%M:%S")
        now_date = time.strftime("%Y/%m/%d")
    
        self.week_day_dict = {
            0 : '(星期一)',
            1 : '(星期二)',
            2 : '(星期三)',
            3 : '(星期四)',
            4 : '(星期五)',
            5 : '(星期六)',
            6 : '(星期天)',
        }
        fontStyle = tkFont.Font(size=14)
        self.day = datetime.date.today().weekday()
        now_date_info = tkinter.Label(text=now_date + ' ' + self.week_day_dict[self.day], 
            bg="red", fg='yellow', font=fontStyle)
        now_date_info.place(x=60, y=80, width=189, height=32)

        now_time = time.strftime("%H:%M:%S")
        fontStyle = tkFont.Font(size=14)
        now_time_info = tkinter.Label(text=now_time, bg="red", fg='yellow', font=fontStyle)
        now_time_info.place(x=254, y=80, width=189, height=32)

        now_hr=int(time.strftime("%H"))
        now_min=int(time.strftime("%M"))
        now_sec=int(time.strftime("%S"))
        
        self.window.after(500, self.update_clock)

    def Face_detection(self,frame):
        frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.9:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, 
                # then pass the blob through our face embedding model to 
                # obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                # perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                self.name = self.le.classes_[j]

                # draw the bounding box of the face along with the associated probability
                face_text = "{}: {:.2f}%".format(self.name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10

                self.fontStyle = tkFont.Font(size=12)
                self.face_name_label = tkinter.Label(text=face_text, bg="blue", fg='white', font=self.fontStyle)
                self.face_name_label.place(x=40, y=320, width=189, height=24)

    def load_labels(self,path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}

    def set_input_tensor(self,interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image


    def classify_image(self,interpreter, image, top_k=1):
        """Returns a sorted array of classification results."""
        self.set_input_tensor(interpreter, image)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        # If the model is quantized (uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)

        ordered = np.argpartition(-output, top_k)
        return [(i, output[i]) for i in ordered[:top_k]]

    def classification(self,frame):
        frame_width=frame.shape[1]
        frame_height=frame.shape[0]

        cut_d=int((frame_width-frame_height)/2)
        crop_img=frame[0:frame_height,cut_d:(cut_d+frame_height)]

        image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

        start = time.perf_counter()
        results = self.classify_image(self.interpreter, image)
        label_id, prob = results[0]
        self.inference_time = time.perf_counter() - start

        # print(labels[label_id],prob)
        #cv2.putText(crop_img,labels[label_id] + " " + str(round(prob,3)) 
            #+ " Inference time=" + str(round(inference_time*1000,2)) + "ms", 
            #(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

        self.image_text = "{}: {:.2f}%".format(self.labels[label_id], prob * 100)

        self.fontStyle = tkFont.Font(size=12)
        self.image_label = tkinter.Label(text=self.image_text, bg="blue", fg='white', font=self.fontStyle)
        self.image_label.place(x=40, y=320, width=189, height=24)


class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.hint_teacher_check=0

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if (self.vid.isOpened()):
            self.vid.release()
        self.vid.release()


# Create a window and pass it to the Application object
App(tkinter.Tk(), '影像識別 GUI設計')
