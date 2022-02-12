
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import tkinter as tk
from PIL import ImageTk

import tensorflow.compat.v1 as tf
from imutils.video import VideoStream
import re
from os import listdir
from os.path import isfile, join
import argparse
import face_net as facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import imageio
import configparser
import json
import tkinter
from tkinter import*
from tkinter.ttk import *
import PIL.Image
# from tkinter import * from tkinter.ttk import *

# window=tk.Tk()
# window.geometry("620x780")
# pane = Frame(window)
# window.title("Face Recognition")

# pane.pack(fill = "both", expand = True)
# image=PIL.Image.open('Dataset/FaceData/raw/Mr Tung/tung.jpg')
# image.thumbnail((300,300),PIL.Image.ANTIALIAS)
# photo=ImageTk.PhotoImage(image)
# panel = tk.Label(window, image=photo)

input_list = []

conf = configparser.ConfigParser()
conf.read('config.ini')
min_size = int(conf['Main']['MINSIZE'])
thres_hold = json.loads(conf['Main']['THRESHOLD'])
factor = float(conf['Main']['FACTOR'])
input_image_size = int(conf['Main']['INPUT_IMAGE_SIZE'])
classifier_path = conf['Main']['CLASSIFIER_PATH']
facenet_model_path = conf['Main']['FACENET_MODEL_PATH']
folder_test = conf['Main']['FOLDER_TEST']

result_file = "{}/result.txt".format(folder_test)


THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
dataset = facenet.get_dataset(folder_test)
def test_face_folder():
        
        # load model trained
    # with open(classifier_path, 'rb') as file:
    #     model, class_names = pickle.load(file)

    # #load tf session
    # with tf.Graph().as_default():

    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    #     with sess.as_default():
    #         # Load the model
    #         print('Loading feature extraction model')
    #         facenet.load_model(FACENET_MODEL_PATH)

    #          # Load the model
    #         facenet.load_model(facenet_model_path)
    #         # Get input and output tensors
    #         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
    #         person_detected = collections.Counter()
            # facenet.load_model(facenet_model_path)
            # # Get input and output tensors
            # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
            # person_detected = collections.Counter()
    for cls in dataset:
        for image_path in cls.image_paths:
            info_path = re.split(r' |/|\\' , image_path)
            folder_result = "{}/{}_{}".format(info_path[0], 'result', info_path[1])
            isExist = os.path.exists(folder_result)
            if not isExist:
                os.makedirs(folder_result)
            path_result = "{}/{}".format(folder_result, info_path[-1])
            try:
                if choosecamera == True:
                    print("choosecamera")
                    img = cap.read()
                    img = imutils.resize(img, width=300)
                    frame = cv2.flip(img, 1)
                else:
                    img = imageio.imread(image_path)
                    if img.ndim<2:
                        print('Unable to align "%s"' %image_path)
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = imutils.resize(img, width=300)
                    frame = img[:,:,0:3]
                bounding_boxes, _ = align.detect_face.detect_face(frame, min_size, pnet, rnet, onet, thres_hold, factor)
                faces_found = bounding_boxes.shape[0]
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                print("photo",photo)
                canvas.create_image(0,0, image = photo, anchor=tkinter.NW)
                # try:
                #     if faces_found > 0:
                #         det = bounding_boxes[:, 0:4]
                #         bb = np.zeros((faces_found, 4), dtype=np.int32)
                        
                #         for i in range(faces_found):
                #             bb[i][0] = det[i][0]
                #             bb[i][1] = det[i][1]
                #             bb[i][2] = det[i][2]
                #             bb[i][3] = det[i][3]
                #             if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                #                 cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                #                 scaled = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                #                 scaled = facenet.prewhiten(scaled)
                #                 scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                #                 feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                #                 emb_array = sess.run(embeddings, feed_dict=feed_dict)
                #                 predictions = model.predict_proba(emb_array)
                #                 best_class_indices = np.argmax(predictions, axis=1)
                #                 best_class_probabilities = predictions[
                #                     np.arange(len(best_class_indices)), best_class_indices]
                #                 best_name = class_names[best_class_indices[0]]
                #                 cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                #                 text_x = bb[i][0]
                #                 text_y = bb[i][3] + 20
                #                 if best_class_probabilities > 0.8:
                #                     name = class_names[best_class_indices[0]]
                #                     person_detected[best_name] += 1
                #                 else:
                #                     name = "Unknown"
                #                 cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                #                 cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 255, 255), thickness=1, lineType=2)
                #                 text =  "{} ---> [ Name: {}, Probability: {} ] ".format(image_path, best_name, best_class_probabilities)
                #                 print("text", text)
                #                 with open(result_file, mode='a+', encoding='utf-8') as f:
                #                     f.write(text + '\n')
                #                 #     f.close()
                #                 # f.write(text)
                #         # cv2.imshow('Face Recognition', frame)
                #         cv2.imwrite(path_result, frame)
                #         img = ImageTk.PhotoImage(PIL.Image.open(path_result))
                #         # label_image.configure(image = frame)
                #         canvas.configure(image=img)
                #         canvas.image = frame
                # except:
                #     pass
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
    print("Done!!!")
    window.after(5, test_face_folder)

    # load model trained
if __name__ == "__main__":

      
        # load model trained
    with open(classifier_path, 'rb') as file:
        model, class_names = pickle.load(file)

    #load tf session
    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

             # Load the model
            facenet.load_model(facenet_model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
            person_detected = collections.Counter()
            cap  = VideoStream().start()

            #create Tkinter
            window = Tk()
            window.title("Hệ thống nhận diện khuôn mặt mở cửa")

            canvas = Canvas(window, width = 300, height= 230)
            canvas.pack()
            button = Button(window,text = "Test face recognition", command=test_face_folder)
            button.pack()
            photo = None
            choosecamera = False
            #Distance = int(input('distance: '))
            # while True:
            #     if choosecamera:
            #         test_face_folder()
            #     else:
            #         print("Error")
            window.mainloop()