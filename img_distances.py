
"""
Feature extractor and KNN model training routine. 
----
Part of the image retrieval solution proposed by the Roosters team,
AML course, University of Trento, May 2021. 
"""

from parse_images import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import tqdm
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans

import cv2
import os

class FeatureExtractor(): 

    def __init__(self, feature_extractor, model, out_dim=20, scale=None,
                 subsample=100):

        self.feature_extractor = feature_extractor
        self.model = model
        self.scale = scale
        self.subsample = subsample

    def get_descriptor(self, img_path): #broken
        """
        Returns descriptor vector associated to a given image 
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs


    def fit_model(self, data_list):
        training_feats = []
        # we extact ORB descriptors
        for img_path in  data_list: #tqdm(data_list, desc='Fit extraction'): #data_list is enumerable containing paths
            print('Analyzing image at location: {}'.format(img_path))
            descs = self.get_descriptor(img_path)
            if descs is None:
                continue
            
            if self.subsample:
                sub_idx = np.random.choice(np.arange(descs.shape[0]), self.subsample)
                descs = descs[sub_idx, :]

            training_feats.append(descs)
        training_feats = np.concatenate(training_feats)
        print('--> Model trained on {} features'.format(training_feats.shape))
        # we fit the model
        self.model.fit(training_feats)
        print('--> Model fit')


    def fit_scaler(self, data_list):
        features = self.extract_features(data_list)
        print('--> Scale trained on {}'.format(features.shape))
        self.scale.fit(features)
        print('--> Scale fit')


    def extract_features(self, data_list):
        # we init features
        features = np.zeros((len(data_list), self.model.n_clusters))
        i=-1
        for img_path in data_list: #enumerate(tqdm(data_list, desc='Extraction')):
            print('Analyzing image at location: {}'.format(img_path))
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


