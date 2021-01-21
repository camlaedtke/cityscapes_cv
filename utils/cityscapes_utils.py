import os
import sys
import cv2
import PIL
import json
import glob
import random
import imageio
import sklearn
import itertools
import numpy as np

from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                              
    for subdir in subdirs:                                                                                     
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                       
                r.append(os.path.join(subdir, file))                                                                       
    return r 


def populate_directory(input_dir, annotation_dir, image_dir):
    
    ids_temp = list_files(input_path)
    
    mask_list = []
    for i in ids_temp:
        if i.endswith("labelIds.png"):
            mask_list.append(i)
            
    image_list = []
    for i in ids_temp:
        if i.endswith("leftImg8bit.png"):
            image_list.append(i)
            
                    
    for n, mask_id in enumerate(mask_list):
        img = load_img(mask_id, color_mode = "grayscale")
        print("\r saving {} / {}".format(n+1, len(mask_list)), end='')
        
        id_temp = mask_id.split("\\")
        id_temp = id_temp[-1]
        
        img.save(annotation_dir + id_temp)
        
        
    for n, img_id in enumerate(image_list):
        img = load_img(img_id, color_mode = "rgb")
        print("\r saving {} / {}".format(n+1, len(image_list)), end='')
        
        id_temp = img_id.split("\\")
        id_temp = id_temp[-1]
        
        img.save(image_dir + id_temp)
        
    print("\n done!")
    
    
    
def get_images(path, img_height, img_width, coarse=True, subset=None):
    ids_temp = next(os.walk(path + "annotations"))[2]
    ids_1 = []
    for i in ids_temp:
        if i.endswith("labelIds.png"):
            id_temp = i.split("\\")
            if coarse:
                id_temp = id_temp[-1][:-22]
            else:
                id_temp = id_temp[-1][:-20]
            ids_1.append(id_temp)
            
    random.seed(2019)
    id_order = np.arange(len(ids_1))
    np.random.shuffle(id_order)
    
    ids = []
    for i in range(len(id_order)):
        ids.append(ids_1[np.int(id_order[i])])
        
    if (subset is not None):
        X = np.zeros((subset, img_height, img_width, 3), dtype=np.float32)
        print("Number of images: " + str(subset))
    else:
        X = np.zeros((len(ids), img_height, img_width, 3), dtype=np.float32)
        print("Number of images: " + str(len(ids)))
        
    for n, id_ in enumerate(ids):
        print("\r Loading %s \ %s " % (n+1, len(ids)), end='')
        
        id_image = id_ + "_leftImg8bit.png"
        image_filename = path + "images\\" + id_image

        # load image
        img = load_img(image_filename)
        x_img = img_to_array(img)
        x_img = resize(x_img, (img_height, img_width, 3), mode='constant', preserve_range = True)
        # save image
        X[n, ...] = x_img.squeeze()
        
        if (subset is not None) and (n == subset-1):
            break
            
    print("Done!")
    return np.array(X)
    
    
def dump_rgb_data(X_all, fp):
    
    R_MEAN = np.mean(X_all[:,:,:,0])
    G_MEAN = np.mean(X_all[:,:,:,1])
    B_MEAN = np.mean(X_all[:,:,:,2])
    
    print("Mean value of first channel: {}".format(R_MEAN))
    print("Mean value of second channel: {}".format(G_MEAN))
    print("Mean value of third channel: {}".format(B_MEAN))
    
    R_STD = np.std(X_all[:,:,:,0])
    G_STD = np.std(X_all[:,:,:,1])
    B_STD = np.std(X_all[:,:,:,2])
    
    print("Std of first channel: {}".format(R_STD))
    print("Std of second channel: {}".format(G_STD))
    print("Std of third channel: {}".format(B_STD)) 
    
    info_dict = {
        "R_MEAN": str(R_MEAN),
        "G_MEAN": str(G_MEAN),
        "B_MEAN": str(B_MEAN),
        "R_STD": str(R_STD),
        "G_STD": str(G_STD),
        "B_STD": str(B_STD),
    }
    
    # Opening JSON file, write to JSON file
    with open(fp, 'w') as openfile: 
        # Reading from json file 
        json.dump(info_dict, openfile) 
    
    
    
def normalize_channels(X_train, X_test):
    
    R_MEAN = np.mean(X_train[:,:,:,0])
    G_MEAN = np.mean(X_train[:,:,:,1])
    B_MEAN = np.mean(X_train[:,:,:,2])
    
    print("Mean value of first channel: {}".format(R_MEAN))
    print("Mean value of second channel: {}".format(G_MEAN))
    print("Mean value of third channel: {}".format(B_MEAN))
    
    R_STD = np.std(X_train[:,:,:,0])
    G_STD = np.std(X_train[:,:,:,1])
    B_STD = np.std(X_train[:,:,:,2])
    
    print("Std of first channel: {}".format(R_STD))
    print("Std of second channel: {}".format(G_STD))
    print("Std of third channel: {}".format(B_STD))
    
    X_train[:,:,:,0] -= R_MEAN
    X_train[:,:,: 1] -= G_MEAN
    X_train[:,:,: 2] -= B_MEAN
    
    X_train[:,:,:,0] /= R_STD
    X_train[:,:,: 1] /= G_STD
    X_train[:,:,: 2] /= B_STD
    
    X_test[:,:,:,0] -= R_MEAN
    X_test[:,:,: 1] -= G_MEAN
    X_test[:,:,: 2] -= B_MEAN
    
    X_test[:,:,:,0] /= R_STD
    X_test[:,:,: 1] /= G_STD
    X_test[:,:,: 2] /= B_STD
    
    return X_train, X_test