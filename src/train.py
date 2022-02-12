from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
import argparse
# import facenet1 as facenet
import face_net as facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

def main():
    # dat bien dieu kien
    batch_size = 1000
    image_size = 160
    batch_size=1000
    classifier_filename="Models/facemodel.pkl"
    model="Models/20180402-114759.pb"
    data_dir="Dataset/FaceData/processed" # du lieu anh de train - mat da duoc cat
    image_size=160
    # min_nrof_images_per_class=20
    # nrof_train_images_per_class=10
    seed=666
    # test_data_dir=None
    # use_split_dataset=False

    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=seed)
            print("data_dir", data_dir)
            dataset = facenet.get_dataset(data_dir) # lay dataset 
            print("dataset", dataset)
            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
            paths, labels = facenet.get_image_paths_and_labels(dataset) # lay anh _ label: 123_12.png tung
            print("paths, labels", paths, labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model) # load model de train
            
            # Get input and output tensors tensflow - 
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))# tao mang nhung, chuoi numpy, 
            print("emb_array", emb_array)

            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                print("emb_array[start_index:end_index,:]", start_index, end_index, emb_array[start_index:end_index,:])
                print("len emb_array", len(emb_array))
            print("emb_array", emb_array)
            
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            # Train classifier TRAIN
            print('Training classifier')
            model = SVC(kernel='linear', probability=True) #linear
            model.fit(emb_array, labels)
            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    for cls in dataset:
        paths = cls.image_paths
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
    return train_set

            

if __name__ == '__main__':
    main()
