from keras.utils import to_categorical 
import pandas as pd 
import numpy as np 
from PIL import Image 
import tensorflow as tf 


class AydFashionDataGenerator(): 
    """
        Data generator for the Aydinli Fashion Items. 
        This class can be used when training our Keras multi-output model. 
    """
    
    def __init__(self, Xy_train: pd.DataFrame, Xy_test: pd.DataFrame, shape=(224, 224)) -> None: 
        self.Xy_train = Xy_train 
        self.Xy_test = Xy_test 
        self.shape = shape 
        
        merged = pd.concat([Xy_train, Xy_test], axis=0) 
        self.yakaTipi_arr_max = merged["YakaTipi_id"].value_counts().index.max()
        self.CepOzelligi_arr_max = merged["CepOzelligi_id"].value_counts().index.max()
        self.KolBoyuAciklama_arr_max = merged["KolBoyuAciklama_id"].value_counts().index.max()


    
    def generate_split_indexes(self):  
         pass 

    def cast_id_2_one_hot(self, arr: np.array, ds_max: int): 
        zeros = np.zeros((arr.size, ds_max + 1)) 
        zeros[np.arange(arr.size), arr] = 1
        return zeros
    
    def preprocess_image(self, img_path):
        """
            Used to perform some minor preprocessing on the image before inputting into the network.
        """  
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img.numpy() 
    
    def generate_images(self, is_training: bool, bathc_size=16): 
        """
            Used to generate a batch with images when training/testing/validating our Keras model.  
        """
        # arrays to store our batched data 
        images, attr_yaka, attr_kolBoyu, attr_cep = [], [], [], [] 
        
        if is_training:
            Xy = self.Xy_train 
        else:
            Xy = self.Xy_test 

        while True:
            for idx, row in Xy.iterrows():
                im = self.preprocess_image(row["paths"]) 
                id_yaka = row["YakaTipi_id"]
                id_cep = row["CepOzelligi_id"]
                id_kolBoyu = row["KolBoyuAciklama_id"] 

                images.append(im) 
                attr_yaka.append(id_yaka) 
                attr_cep.append(id_cep) 
                attr_kolBoyu.append(id_kolBoyu) 

                if len(images) >= bathc_size:
                    yield np.array(images),  {"YakaTipi": self.cast_id_2_one_hot(np.array(attr_yaka), self.yakaTipi_arr_max), 
                                             "CepOzelligi": self.cast_id_2_one_hot(np.array(attr_cep), self.CepOzelligi_arr_max), 
                                             "KolBoyuAciklama": np.array(attr_kolBoyu)
                                             }
                    images, attr_yaka, attr_kolBoyu, attr_cep = [], [], [], [] 

            if not is_training:
                break 

class MultiOutputDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator): 
    def flow(self, 
             x, 
             y=None, 
             batch_size=32, 
             shuffle=True, 
             sample_weight=None, 
             seed=None, 
             save_to_dir=None, 
             save_prefix="", 
             save_format="png", 
             ignore_class_split=False, 
             subset=None): 
        
        targets = None 
        target_lengths = {} 
        ordered_outputs = [] 
        for output, target in y.items(): 
            if targets is None:
                targets = target  
            else: 
                targets = np.concatenate((targets, target), axis=1) 
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output) 
        
        # for flowX, flowy in super().flow(x, targets, batch_size=batch_size, shuffle=shuffle,seed=seed): 


    def flow_from_dataframe(self, 
                            dataframe, 
                            directory=None, 
                            x_col="filename", 
                            y_col="class", 
                            weight_col=None, 
                            target_size=..., 
                            color_mode="rgb", 
                            classes=None, 
                            class_mode="categorical", 
                            batch_size=32, 
                            shuffle=True, 
                            seed=None, 
                            save_to_dir=None, 
                            save_prefix="", 
                            save_format="png", 
                            subset=None, 
                            interpolation="nearest", 
                            validate_filenames=True, 
                            **kwargs):
        
        
        
        
        return super().flow_from_dataframe(dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)
    






             


                

        

