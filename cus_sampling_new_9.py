### Using X_test, Select 50% point near center and 50% far from center in total point which is 50% of majority class & Neighbourhood Cleaning
# A, B, C

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import interp
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold

def distance(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a-b)

def cus_sampler_new_9(X_train, y_train, X_test, number_of_clusters=23, percentage_to_choose_from_each_cluster=0.50):
    """
    number_of_clusters = 23
    percentage_to_choose_from_each_cluster: 50%
    """

    selected_idx = []
    selected_idx = np.asarray(selected_idx)

    value, counts = np.unique(y_train, return_counts=True)
    minority_class = value[np.argmin(counts)]
    majority_class = value[np.argmax(counts)]

    idx_min = np.where(y_train == minority_class)[0]
    idx_maj = np.where(y_train == majority_class)[0]

    majority_class_instances = X_train[idx_maj]
    majority_class_labels = y_train[idx_maj]
    minority_class_instances = X_train[idx_min]
    minority_class_labels = y_train[idx_min]
    
#     print(len(majority_class_labels))
#     print(len(minority_class_labels))
    
    total_X = np.append(majority_class_instances, X_test, axis = 0)

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(total_X)

    X_maj = []
    y_maj = []
    
    clus = kmeans.cluster_centers_
    X_train_predict = kmeans.predict(X_train)

    points_under_each_cluster = {i: np.where(X_train_predict == i)[0] for i in range(kmeans.n_clusters)}

    selected_majority_idx = list()

    for key in points_under_each_cluster.keys():
        points_under_this_cluster = np.array(points_under_each_cluster[key])

    #     print(points_under_this_cluster)

        minority_point_under_this_cluster = []
        majority_point_under_this_cluster = []

        for i in points_under_this_cluster:
            if i in idx_min:
                minority_point_under_this_cluster.append(i)
            else:
                majority_point_under_this_cluster.append(i)
#         print(minority_point_under_this_cluster)
#         print(majority_point_under_this_cluster)
#         print("......")

        number_of_points_to_choose_from_this_cluster = math.ceil(
            len(majority_point_under_this_cluster) * percentage_to_choose_from_each_cluster)
        
        
        majority_distance = dict()
        if len(minority_point_under_this_cluster) == 0:
            selected_majority_idx.extend(majority_point_under_this_cluster)
#             print(selected_majority_idx)
#             print("end")
        else:
            for i in minority_point_under_this_cluster:
                for j in majority_point_under_this_cluster:
                    majority_distance[j] = distance(X_train[i], X_train[j])

                sorted_majority_distance = sorted(majority_distance.items(), key = lambda kv:(kv[1], kv[0]))
                number_of_selected_points = int(len(sorted_majority_distance) * percentage_to_choose_from_each_cluster)
                neighborhood_majority_instances = sorted_majority_distance[number_of_selected_points:]
                
                distance_from_cluster_to_each_point = {}
        
                for neighbor in neighborhood_majority_instances:
                    distance_from_cluster_to_each_point[neighbor[0]] = distance(clus[key], neighbor[1])

                sorted_point = sorted(distance_from_cluster_to_each_point.items(), key =lambda kv:(kv[1], kv[0]))
                
                start = math.ceil(len(neighborhood_majority_instances) * 0.5)
                finish = int(len(neighborhood_majority_instances) - start)
                
                
                starting_point = sorted_point[:start]
                finishing_point = sorted_point[-finish:]
                selected_majority_instances = starting_point + finishing_point
                
#                 print("Start: ", len(starting_point))
#                 print("End: ", len(finishing_point))
#                 print("Merge: ", len(selected_majority_instances))
        
#                 selected_majority_instances = np.append(starting_point, finishing_point, axis = 0)
#                 print(starting_point)

                for instance in selected_majority_instances:
                    if instance[0] not in selected_majority_idx:
                        selected_majority_idx.append(instance[0])
#             print("Start...")
#             print(len(sorted_majority_distance))
            
#             print(len(selected_majority_idx))
#             print("End.....")


#     X_sampled = X_train[selected_idx]
#     y_sampled = y_train[selected_idx]

    
    X_sampled = np.concatenate((X_train[idx_min], X_train[selected_majority_idx]))
    y_sampled = np.concatenate((y_train[idx_min], y_train[selected_majority_idx]))

#     print(X_sampled.shape, y_sampled.shape, selected_idx.shape)

    return X_sampled, y_sampled, selected_idx
