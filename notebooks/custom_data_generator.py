from keras.utils import to_categorical 
import pandas as pd 
import numpy as np 
from PIL import Image 

class AydFashionDataGenerator(): 
    """
        Data generator for the Aydinli Fashion Items. 
        This class can be used when training our Keras multi-output model. 
    """
    
    def __init__(self, shuffled_splitted_data, test_size=0.25, shape=(224, 224)) -> None: 
        self.TRAIN_TEST_SPLIT = 1 - test_size 
        self.df = shuffled_splitted_data 
        self.shape = shape 
    
    def generate_split_indexes(self):  
         pass 
    
    def preprocess_image(self, img_path):
        """
            Used to perform some minor preprocessing on the image before inputting into the network.
        """  
        return Image.open(img_path).resize(self.shape) 
        

