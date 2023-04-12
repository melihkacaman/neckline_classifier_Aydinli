from sklearn.metrics import precision_score, recall_score, f1_score 
import tensorflow as tf 
import numpy as np 
import pandas as pd
import cv2 
from matplotlib import pyplot as plt 

class MultiOutputModelTester:
    def __init__(self, model: tf.keras.models.Model, y_test: pd.DataFrame, paths: list, dims: tuple = (224, 224, 3)): 
        """
            model: model which you want to predict with   
            paths: paths of the images 
            dims: default dim of the image 
            y_test: actual results. it has to consist of just ids of outputs and order is important. it must be same with the order of 
                    models outputs layers. 
        """
        self.model = model 
        self.y_test = y_test 
        self.paths = paths 
        self.dims = dims

        self.ys = dict()
        self.y_hats = dict()  
        
        self._predict(self.paths, self.model, self.dims) # to fill y_hats
        self._y_id_to_one_hot(self.y_test) #to fill ys 
        
    def _predict(self, paths: list, model: tf.keras.models.Model, dims: tuple = (224, 224, 3)):
        for col in self.y_test:
            self.y_hats[col] = [] 
                 
        i = 0 
        for path in paths:
            img = tf.io.read_file(path) 
            img = tf.image.decode_jpeg(img, channels=3) 
            img = tf.image.resize(img, [dims[0], dims[1]]) 
            img = tf.reshape(img, [1, dims[0], dims[1], dims[2]])

            d = 0 
            y_hat = model.predict(img, verbose=0)
            for key in self.y_hats.keys():
                self.y_hats[key].append(y_hat[d].squeeze().tolist())
                d += 1 
                    
            i = i + 1 
            if i % 100 == 0: 
                print(f"{i} th iteration. You have {len(paths)} inputs. ")
        
    def _y_id_to_one_hot(self, y_test: pd.DataFrame):
        """           
            return list of one hot dummies for each output
        """
        for col in y_test:
            target =self.y_test[col].to_numpy() 
            if target.max() > 1: 
                self.ys[col] = pd.get_dummies(y_test[col]).to_numpy().tolist()  
            else:
                self.ys[col] = target.tolist()                 
        
    def get_metrics(self, threshold=0.5):
        result = dict() 
        for col in self.y_hats.keys(): 
            custom_yhats = np.array(self.y_hats[col].copy()) 
            if custom_yhats.ndim == 2:
                zeros = np.zeros(custom_yhats.shape) 
                zeros[np.arange(custom_yhats.shape[0]), np.argmax(custom_yhats, axis=1)] = 1
                custom_yhats = zeros 
            else:
                custom_yhats[custom_yhats >= 0.5] = 1 
                custom_yhats[custom_yhats < 0.5] = 0 

            result[col] = {
                "threshold": threshold, 
                "precision_weighted": round(precision_score(self.ys[col], custom_yhats, average="weighted"), 4),
                "recall_weighted": round(recall_score(self.ys[col], custom_yhats, average="weighted"), 4), 
                "f1_score_weighted": round(f1_score(self.ys[col], custom_yhats, average="weighted"), 4)
            }
        return result
    

def custom_predict(model: tf.keras.models.Model, input ,labels: list, classes: dict):
    df = pd.DataFrame() 

    input_x = cv2.resize(input, (224,224))     
    input_x = input_x.reshape(1,224,224,3) 
    result = model.predict(input_x)

    for i, key in enumerate(classes): 
        indexes = [v for v in classes[key].values()] 
        preds_ith = result[i].squeeze() 
        if len(preds_ith.shape) > 0:
            zeros = np.zeros(preds_ith.shape, dtype="int")
            zeros[np.argmax(preds_ith, axis=0)] = 1 
            preds_ith = zeros
        else:
            preds_ith[preds_ith >= 0.5] = 1 
            preds_ith[preds_ith < 0.5] = 0 
            preds_ith.dtype = "int"
            zeros = np.zeros((2), dtype="int")
            zeros[preds_ith] = 1 
            preds_ith = zeros 

        df = pd.concat([
            df,
            pd.DataFrame({
            "classes": indexes, 
            "preds": preds_ith,
            "actuals": labels[i]
        })
        ], axis=0)

    plt.imshow(input / 255.0)
    plt.show()

    return df