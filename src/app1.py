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

window=tk.Tk()
window.geometry("620x780")
pane = Frame(window)
window.title("Face Recognition")

pane.pack(fill = "both", expand = True)
image=PIL.Image.open('Dataset/FaceData/raw/Mr Tung/tung.jpg')
image.thumbnail((300,300),PIL.Image.ANTIALIAS)
photo=ImageTk.PhotoImage(image)
panel = tk.Label(window, image=photo)

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

path = "Dataset/FaceData/raw/Mr Tung/tung.jpg"
path2 = "Dataset/FaceData/raw/Mr Thanh/thanh.jpg"

panel.pack(side="top", fill="both", expand="yes")



def test_face_folder():
       print("tes")
# panel.grid(column=1,row=0)
def callback():
    img = PIL.Image.open(path2)
    print("img", img)
    img2 = ImageTk.PhotoImage(img)
    print("imge", img2)
    panel.configure(image=img2)
    panel.image = img2


button=tk.Button(window,text="Test",command=callback,bg="pink")
button.pack(side="bottom", fill="both", expand="no")
button=tk.Button(window,text="Test folder",command=test_face_folder,bg="pink")
button.pack(side="bottom", fill="both", expand="no")

window.bind("<Return>", callback)
# root.mainloop()

window.mainloop()