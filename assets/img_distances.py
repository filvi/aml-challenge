
"""
Feature extractor and KNN model training routine. 
----
Part of the image retrieval solution proposed by the Roosters team,
AML course, University of Trento, May 2021. 
"""



from assets.dataset import Dataset
from assets.img_manipulation import *
    
import numpy as np
from sklearn.preprocessing import StandardScaler
import tqdm
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
import itertools
import cv2
import os

class FeatureExtractor(): 

    def __init__(self, 
                feature_extractor, 
                model, 
                scale=None,
                subsample=100,
                img_man=False):

        self.feature_extractor = feature_extractor
        self.model = model
        self.scale = scale
        self.subsample = subsample
        self.img_man = img_man

    def get_descriptor(self, 
                       img_path, 
                       img_man=False, 
                       fn=None, 
                       **kwargs):
        """
        Returns descriptor vector associated to a given image 
        :img_man: True if active 
        :fn: The function of img_manipulation to be called one of the following
        ->  pick_color_channel(image, channel)
        ->  noise_over_image(image, prob=0.01)
        ->  fakehdr(image, alpha=-100, beta=355, preset=None)
        ->  enhance_features(img, val1, val2, inverse=True)
        
        :**kwargs: the fn's function parameters from img_manipulation file
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (250,250))

        
        # passing the function to manipulate the image as parameter in get_descriptor
        # passing also eventual switches for the custom function fn called
        if img_man:
            img = fn(img, **kwargs)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs

    
    def fit_model(self, data_list):
        training_feats = []
        
        # We create a list  with all the function to be applied to each image in the dataset to use it later on
        img_man_methods  = [
            pick_red_channel,
            pick_green_channel,
            pick_blue_channel,
            noise_over_image,
            fakehdr,
            enhance_features
            ] 
        
        # we extact ORB descriptors
        for img_path in  data_list: #tqdm(data_list, desc='Fit extraction'): #data_list is enumerable containing paths
            print('Analyzing image at location: {}'.format(img_path), end="\r", flush=True)
            
            # Execute the operation with the original image
            descs = self.get_descriptor(img_path, img_man=False)
            if descs is None:
                continue
            
            if self.subsample:
                sub_idx = np.random.choice(np.arange(descs.shape[0]), self.subsample)
                descs = descs[sub_idx, :]

            training_feats.append(descs)

            # apply each img_manipulation function to the image and append it to training_feats
            for f in img_man_methods:
                # apply for each image each function of img_man_methods
                descs = self.get_descriptor(img_path, img_man=True, fn=f)
                if descs is None:
                    continue
                
                if self.subsample:
                    sub_idx = np.random.choice(np.arange(descs.shape[0]), self.subsample)
                    descs = descs[sub_idx, :]

                training_feats.append(descs)

        training_feats = np.concatenate(training_feats)
        print(" "* 100) # clear the buffer of  end="\r"
        print('--> Model trained on {} features'.format(training_feats.shape))
        # we fit the model
        self.model.fit(training_feats)
        print('--> Model fit')


    def fit_scaler(self, data_list):
        features = self.extract_features(data_list)
        print(" "* 100) # clear the buffer of  end="\r"
        print('--> Scale trained on {}'.format(features.shape))
        self.scale.fit(features)
        print('--> Scale fit')


    def extract_features(self, data_list):
        # we init features
        features = np.zeros((len(data_list), self.model.n_clusters))
        i=-1
        for img_path in data_list: #enumerate(tqdm(data_list, desc='Extraction')):
            print('Analyzing image at location: {}'.format(img_path), end="\r", flush=True)
            i+=1
            # get descriptor
            descs = self.get_descriptor(img_path)
            # 2220x128 descs
            preds = self.model.predict(descs) #returns (2220,) array with Index of the cluster each of the 2220 sample belongs to. 
            histo, _ = np.histogram(preds, bins=np.arange(self.model.n_clusters+1), density=True) #the frequencies of the class values (between 1 and 100) are computed 
            # append histogram
            features[i, :] = histo #you end up having, for each element of the data list, a 100- dim vector containing the probability of each 

        return features

    def scale_features(self, features):
        # we return the normalized features
        return self.scale.transform(features)


def initialize_extractor(training_path):
    feature_extractor = cv2.ORB_create()
    model = KMeans(n_clusters=100, n_init=10, max_iter=5000, verbose=False)
    scale = StandardScaler() 
    extractor = FeatureExtractor(feature_extractor=feature_extractor,
                                model = model,
                                scale = scale)

    trimgs = Dataset(training_path) 
    
    extractor.fit_model(trimgs.get_files())
    extractor.fit_scaler(trimgs.get_files())
    return(extractor) #initialized extractor with fit knn and scaler


