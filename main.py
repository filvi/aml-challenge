# -*- coding: utf-8 -*-
from intro import greetings
import argh
import numpy as np
import cv2
from dataset import dataset 
import os
import img_manipulation
import img_distances
import query
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans 
import plotly as plt
from scipy.spatial import distance

# displays the script title and the names of the partecipants
greetings()

def challenge():
    pass



if __name__ == "__main__":
    challenge()


    # parsing directory and instances of dataset class

    #data_path = '/content/gdrive/MyDrive/dataset'
    data_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/0B5NgX9ua1kQkfmxseGVTVDVuSDROaU1EMFpZUTRvWU9pREx6eXJTSVBHLWFKYmVhT2R6Tjg/Applied Machine Learning LM Data Science/Challenge/dataset'
    training_path = os.path.join(data_path, 'training')

    validation_path = os.path.join(data_path, 'validation')
    gallery_path = os.path.join(validation_path, 'gallery')
    query_path = os.path.join(validation_path, 'query')

    training = dataset(training_path)
    gallery = dataset(gallery_path)
    query = dataset(query_path )

    # print number of files for each dataset
    print(training.len_files())
    print(gallery.len_files())
    print(query.len_files())

    training_path, training_class = training.get_data_paths()

    # we get validation gallery and query data

    gallery_path, gallery_class = gallery.get_data_paths()
    query_path, query_class = query.get_data_paths()


    # initialize ORB 
    ORB = cv2.ORB_create()

    # we read a random image
    img_rgb = cv2.imread(gallery_path[0])

    # convert to grayscale 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    kp = ORB.detect(img_gray,None)

    # get keypoints and descriptors
    kp, descs = ORB.compute(img_gray, None)

    # draw kl
    img_orb=cv2.drawKeypoints(img_gray,kp,img_rgb,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #show img
    plt.imshow(img_orb), plt.show()

    # manipulation of images of training directory
    
    # features extraction

    features = cv2.ORB_create()
    
    # apply MinMaxScaler rescales the data set such that all feature values 
    # are in the range [0, 1]

    scaler = MinMaxScaler(feature_range=(0,1))
   

    # Using KMeans to compute centroids 
    kmeans = KMeans(n_clusters=21, random_state=0)

    # model

    # we define the feature extractor providing the model
    extractor = img_distances.FeatureExtractor(feature_extractor = features,
                                model = kmeans,
                                scale = scaler,
                                out_dim = 21)

    
    # we fit the KMeans clustering model
    extractor.fit_model(training_path)


    # we fit the scaler
    extractor.fit_scaler(training_path)

    #we get query features
    query_features = extractor.extract_features(query_path)
    query_features = extractor.scale_features(query_features)

    # we get gallery features
    gallery_features = extractor.extract_features(gallery_path)
    gallery_features = extractor.scale_features(gallery_features)

    print(gallery_features.shape, query_features.shape)
    mahal_dist = spatial.distance.mahalanobis(query_features, gallery_features)
    print('Mahalanobis distance{}'.format(mahal_dist.shape))

    # we sort matched indices
    indices = np.argsort(mahal_dist, axis=-1)
    gallery_matches = gallery_class[indices]

    # validation


    def topk_accuracy(gt_label, matched_label, k=10):
    # get top-k matches
        matched_label = matched_label[:, :k]

        # init total and correct
        total = matched_label.shape[0]
        correct = 0
        for q_idx, q_lbl in enumerate(gt_label):
            # if any of the top-k label is correct, increase correct
            correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
        acc_tmp = correct/total

        return acc_tmp

    # submit function

    def submit(results, url):
        res = json.dumps(results)
        response = requests.post(url, res)
        result = json.loads(response.text)
        print("accuracy is {}".format(result['results']))


    url = "http://kamino.disi.unitn.it:3001/results/"

    #query result

    group = dict()
    group['groupname'] = "Roosters"

    res = dict()
    
    
    k = 10
    query_arr = [2,8,11]

    for idx in query_arr:
        img_query = cv2.imread(query_path[idx])
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        index = indices[idx]
        head, tail = os.path.split(query_path[idx])
        res[tail] = ""
        for i in range(0, k):
            img_match = cv2.imread(gallery_path[index[i]])
            img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
            head_sim, tail_sim = os.path.split(gallery_path[index[i]])
            res[tail] = tail_sim
            

    group["images"] = res
    group
    submit(group, url)
