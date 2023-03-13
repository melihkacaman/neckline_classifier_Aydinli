import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import requests
import os

def show_image(path, width=224, height=224): 
    img = tf.io.read_file(path) 
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [width, height], antialias=False) 
    img = img / tf.constant([255], dtype=tf.float32) 

    plt.imshow(img) 
    plt.show()

def save_image_cv(url, scale_percent, path, size = None):
    resp = requests.get(url, stream=True).raw 
    
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None: 
        raise TypeError("None type") 
    
    if size == None: 
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dsize = (width, height)
    else: 
        dsize = size 
    
    output = cv2.resize(image, dsize)

    cv2.imwrite(path, output)


def make_dataset_cv(dataset_name, datasource, iteration = None):
    problems_of_idx = []
    iterator = 0 
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.'))
        path = os.path.join('data', dataset_name)
        path = os.path.join(base, path)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        # ../data/dataset_name/
        
        folder_train = os.path.join(path, "train") 
        os.mkdir(folder_train) 
        
        folder_test = os.path.join(path, "test") 
        os.mkdir(folder_test) 

        for img_index, img_row in datasource.iterrows():
                try:
                    tmp = os.path.join(path, img_row["partition"])
                    # data / dataset / train - test / 
                    tmp = os.path.join(tmp, "class_" + str(img_row["class"])) 
                    if not os.path.isdir(tmp): 
                        os.mkdir(tmp) 
                    
                    save_image_cv(img_row['img_url'], 75, os.path.join(tmp, str(img_index) + '.png'), size=(224, 224))
                    if iteration is not None and iterator == iteration:
                        break
                    else:
                        iterator+=1
                        if iterator % 1000 == 0: 
                            print(f"Iteration {iterator}") 
                            
                except Exception as e2:
                    problems_of_idx.append(img_index)
                    print(e2)
                    continue
    except Exception as e:
        print('An exception occurred.', e)
    return problems_of_idx