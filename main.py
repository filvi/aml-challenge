# -*- coding: utf-8 -*-
# %%
from intro import greetings
import argh
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
import plotly as plt
from scipy.spatial import distance

# %%
# displays the script title and the names of the partecipants
greetings()



# parsing directory and instances of dataset class

# %%

data_path = 'dataset' # TODO dare la scelta all'utente di caricare il path che vuole
training_path = os.path.join(data_path, 'training')

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


features = cv2.ORB_create() # TODO questo pezzo e' stato estratto dal codice commentato
kmeans = KMeans(n_clusters=100, n_init=10, max_iter=5000, verbose=False) #KMeans(n_clusters=21, random_state=0)
scaler = StandardScaler() #MinMaxScaler(feature_range=(0,1))


len(gallery.get_files())
# %%
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

# TODO mettiamo uno switch per il debug altrimenti lo togliamo
print(" "* 100) # clear the buffer of  end="\r"
print(f"gallery feature shape:{gallery_features.shape}, query features shape:{query_features.shape}")
mahal_dist = distance.cdist(query_features, gallery_features, 'mahalanobis')
print('Mahalanobis distance{}'.format(mahal_dist.shape))
# %%

# we sort matched indices
indices = np.argsort(mahal_dist, axis=-1)
#gallery_matches = gallery_class[indices]


# ottimo per la submission
matches = dict()
for i in range(indices.shape[0]):
    gallery_matches = []
    
    for j in range(indices.shape[1]):
        
        img_path = all_gallery_path[indices[i][j]]  
        img_name = img_path.split(os.path.sep)[-1]  
        gallery_matches.append(img_name) #append the image names in an order that reflects the distance with the i th qry image

        query_img_name = all_query_path[i]
        query_img_name = query_img_name.split(os.path.sep)[-1]  



    matches[query_img_name] = gallery_matches
# %%
gallery_classes = gallery.get_class()
query_classes = gallery.get_class()
finalmatrix = np.zeros(indices.shape)

"""
for i in range(indices.shape[0]):#iterate over rows: qry
    gallery_matches = []
    for j in range(indices.shape[1]):#iterate over columns
        # Get only the topmost matches

        gall_img_class = gallery_classes[indices[i][j]]  
        gallery_matches.append(gall_img_class)


    # if finalmatrix.size == 0:
    if i == 0:
        finalmatrix = np.array(gallery_matches)  #append here the line to the matrix
    else:
        finalmatrix =np.vstack(gallery_matches)
"""   

for i in range(indices.shape[0]):#iterate over rows: qry
    gallery_matches = []
    for j in range(indices.shape[1]):#iterate over columns
        finalmatrix[i][j] = gallery_classes[indices[i][j]]  




# %%
# %%

    
def topk_accuracy(gt_label, matched_label, k=1):
    matched_label = matched_label[:, :k]
    total = matched_label.shape[0]
    correct = 0
    for q_idx, q_lbl in enumerate(gt_label):
        correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
    acc_tmp = correct/total

    return acc_tmp

print('########## RESULTS ##########')

for k in [1, 3, 10]:
    topk_acc = topk_accuracy(query_classes, gallery_matches, k)
    print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))

# %%

    

    # validation


    # def topk_accuracy(gt_label, matched_label, k=10):
    # # get top-k matches
    #     matched_label = matched_label[:, :k]

    #     # init total and correct
    #     total = matched_label.shape[0]
    #     correct = 0
    #     for q_idx, q_lbl in enumerate(gt_label):
    #         # if any of the top-k label is correct, increase correct
    #         correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
    #     acc_tmp = correct/total

    #     return acc_tmp

    # # submit function

    # def submit(results, url):
    #     res = json.dumps(results)
    #     response = requests.post(url, res)
    #     result = json.loads(response.text)
    #     print("accuracy is {}".format(result['results']))


    # url = "http://kamino.disi.unitn.it:3001/results/"

    # #query result

    # group = dict()
    # group['groupname'] = "Roosters"

    # res = dict()


    # k = 10
    # query_arr = [2,8,11]

    # for idx in query_arr:
    #     img_query = cv2.imread(all_query_path[idx])
    #     img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    #     index = indices[idx]
    #     head, tail = os.path.split(all_query_path[idx])
    #     res[tail] = ""
    #     for i in range(0, k):
    #         img_match = cv2.imread(all_gallery_path[index[i]])
    #         img_match = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
    #         head_sim, tail_sim = os.path.split(all_gallery_path[index[i]])
    #         res[tail] = tail_sim


    # group["images"] = res
    # group
    # submit(group, url)



# TODO decommentami
# if __name__ == "__main__":
#     challenge()