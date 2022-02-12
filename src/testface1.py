# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

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
import datetime
import tkinter as tk
from PIL import Image,ImageTk

class RecognitionFace:
    def __init__(self):
        self.input_list = []

        conf = configparser.ConfigParser()
        conf.read('config.ini')
        self.min_size = int(conf['Main']['MINSIZE'])
        self.thres_hold = json.loads(conf['Main']['THRESHOLD'])
        self.factor = float(conf['Main']['FACTOR'])
        self.input_image_size = int(conf['Main']['INPUT_IMAGE_SIZE'])
        self.classifier_path = conf['Main']['CLASSIFIER_PATH']
        self.facenet_model_path = conf['Main']['FACENET_MODEL_PATH']
        self.folder_test = conf['Main']['FOLDER_TEST']
    
    def run(self):
        self.result_file = "{}/result.txt".format(self.folder_test)

        with open(self.classifier_path, 'rb') as file:
            model, class_names = pickle.load(file)
        dataset = facenet.get_dataset(self.folder_test)
        with tf.Graph().as_default():

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default():

                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(self.facenet_model_path)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
                person_detected = collections.Counter()
                for cls in dataset:
                    for image_path in cls.image_paths:
                        info_path = re.split(r' |/|\\' , image_path)
                        folder_result = "{}/{}_{}".format(info_path[0], 'result', info_path[1])
                        isExist = os.path.exists(folder_result)
                        if not isExist:
                            os.makedirs(folder_result)
                        path_result = "{}/{}".format(folder_result, info_path[-1])
                        print("image test: ", image_path)
                        try:
                            img = imageio.imread(image_path)
                            if img.ndim<2:
                                print('Unable to align "%s"' %image_path)
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            frame = img[:,:,0:3]
                            bounding_boxes, _ = align.detect_face.detect_face(frame, self.min_size, pnet, rnet, onet, self.thres_hold, self.factor)
                            faces_found = bounding_boxes.shape[0]
                            try:
                                if faces_found > 0:
                                    det = bounding_boxes[:, 0:4]
                                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                                    print(" range(faces_found)",  range(faces_found))
                                    
                                    for i in range(faces_found):
                                        print("i face", i)
                                        bb[i][0] = det[i][0]
                                        bb[i][1] = det[i][1]
                                        bb[i][2] = det[i][2]
                                        bb[i][3] = det[i][3]
                                        if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                            scaled = cv2.resize(cropped, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
                                            scaled = facenet.prewhiten(scaled)
                                            scaled_reshape = scaled.reshape(-1, self.input_image_size, self.input_image_size, 3)
                                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                                            predictions = model.predict_proba(emb_array)
                                            best_class_indices = np.argmax(predictions, axis=1)
                                            best_class_probabilities = predictions[
                                                np.arange(len(best_class_indices)), best_class_indices]
                                            best_name = class_names[best_class_indices[0]]
                                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                            text_x = bb[i][0]
                                            text_y = bb[i][3] + 20
                                            if best_class_probabilities > 0.8:
                                                name = class_names[best_class_indices[0]]
                                                person_detected[best_name] += 1
                                            else:
                                                name = "Unknown"
                                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 255, 255), thickness=1, lineType=2)
                                            text =  "{} ---> [ Name: {}, Probability: {} ] ".format(image_path, best_name, best_class_probabilities)
                                            print(text)
                                            with open(self.result_file, mode='a+', encoding='utf-8') as f:
                                                f.write(text + '\n')
                                            #     f.close()
                                            # f.write(text)
                                    # cv2.imshow('Face Recognition', frame)
                                    cv2.imwrite(path_result, frame)
                            except:
                                pass
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
        return True
