# -*- coding: utf-8 -*-
from intro import greetings
import numpy as np
import cv2
from dataset import Dataset 
import os
import img_manipulation
import img_distances
import query 
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pprint import pprint
from sklearn.cluster import KMeans 

from scipy.spatial import distance
import time
import requests
# displays the script title and the names of the partecipants
greetings()



# parsing directory and instances of dataset class

start = time.time()
def challenge():
    data_path = 'exam' 
    # training_path   = os.path.join(data_path, 'training')
    training_path   = os.path.join(data_path, 'validation', 'gallery')

    validation_path = os.path.join(data_path, 'validation')
    gallery_path    = os.path.join(validation_path, 'gallery')
    query_path      = os.path.join(validation_path, 'query')


    training = Dataset(os.walk(training_path, topdown=False))
    gallery  = Dataset(os.walk(gallery_path,  topdown=False))
    query    = Dataset(os.walk(query_path,    topdown=False))

    # print number of files for each dataset


    all_training_path = training.get_files()
    training_class = training.get_class() 


    # we get validation gallery and query data
    all_gallery_path = gallery.get_files()
    gallery_class = gallery.get_class() 

    all_query_path = query.get_files()
    query_class = query.get_class() 


    features = cv2.ORB_create() 
    kmeans = KMeans(n_clusters=100, n_init=10, max_iter=5000, verbose=False) 
    scaler = StandardScaler() 


    len(gallery.get_files())
    # we define the feature extractor providing the model
    extractor = img_distances.FeatureExtractor(feature_extractor = features,
                                model = kmeans,
                                scale = scaler)


    # we fit the KMeans clustering model
    extractor.fit_model(all_training_path)


    # we fit the scaler
    extractor.fit_scaler(all_training_path)

    #we get query features
    query_features = extractor.extract_features(all_query_path)
    query_features = extractor.scale_features(query_features)

    # we get gallery features
    gallery_features = extractor.extract_features(all_gallery_path)
    gallery_features = extractor.scale_features(gallery_features)

    print(" "* 100) # clear the buffer of  end="\r"
    print(f"gallery feature shape:{gallery_features.shape}, query features shape:{query_features.shape}")
    mahal_dist = distance.cdist(query_features, gallery_features, 'mahalanobis')
    print('Mahalanobis distance{}'.format(mahal_dist.shape))

    # we sort matched indices
    indices = np.argsort(mahal_dist, axis=-1)
    gallery_matches = gallery_class[indices]
    
    def topk_accuracy(gt_label, matched_label, k=1):

        matched_label = matched_label[:, :k]
        total = matched_label.shape[0]
        correct = 0
        for idx, label in enumerate(gt_label):
            correct+= np.any(label == matched_label[idx, :]).item()
        acc_tmp = correct/total

        return acc_tmp

    print('########## Accuracy ##########')
    
    for k in [1, 3, 10]:
        topk_acc = topk_accuracy(query_class, gallery_matches, k)
        print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))


    group = dict()
    group['groupname'] = "Roosters"
    query_arr = []
    # ottimo per la submission
    matches = dict()
    query_index = [2,3]
    top_k = 10
    for i in range(indices.shape[0]):
    # for i in query_index:
        gallery_matches = []
        for j in range(0,top_k):
            img_path = all_gallery_path[indices[i][j]]
            img_name = img_path.split(os.path.sep)[-1]
            gallery_matches.append(img_name) #append the image names in an order that reflects the distance with the i th qry image
            query_img_name = all_query_path[i]
            query_img_name = query_img_name.split(os.path.sep)[-1]
        matches[query_img_name] = gallery_matches


    group["images"] = matches
    
    # submit function
    
    def submit(results, url):
        res = json.dumps(results)
        response = requests.post(url, res)
        result = json.loads(response.text)
        print("accuracy is {}".format(result['results']))



    url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/competition/"
    
    
    # submit(group, url) 

    



if __name__ == "__main__":
    challenge()
    end = time.time()
    print(f"total time: {end - start}")
