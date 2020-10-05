from sklearn.cluster import KMeans
from utils import configs
from scipy.stats import mode
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np


dataset = configs.FILES.API

embedding_file = os.path.join(dataset,'gcn_node_embeddings0.3_0.txt')
labels_file = os.path.join(dataset, "labels0.txt")
top20category_file = os.path.join(dataset, 'top20category.txt')
categoryNames = os.path.join(dataset, 'categoryNames0.txt')
apimashup_ids = os.path.join(dataset, 'apimashup_ids0.txt')


idenmap = {}
with open(apimashup_ids, 'r') as ar:
    for line in ar:
        ps = line.split()
        idenmap[ps[0]] = ps[1]


cats20 = []
with open(top20category_file, 'r') as tr: 
    for line in tr:
        ps = line.split()
        cats20.append(ps[0])
        
cats = []
with open(categoryNames, 'r') as cr:
    for line in cr:
        ps = line.split()
        if ps[1] in cats20:
            cats.append(ps[0])

label_map = {}
with open(labels_file) as lr:
    for line in lr:
        params = line.split()
        node_id = params[0]
        
        if params[1] in cats:
            labelss = int(params[1].replace('L',''))
            label_map[node_id] = labelss
        
groups = list(set(label_map.values()))
groups.sort()
# 
# 
embeds_m = []
labels_m = []
embeds_a = []
labels_a = []
with open(embedding_file) as er:
    er.readline()
    for line in er:
        params = line.split()
        if not params[0] in label_map.keys():
            continue
        
        if idenmap[params[0]] == 'A':
            embeds_a.append(params[1:])
            labels_a.append(label_map[params[0]]) 
        if idenmap[params[0]] == 'M':
            embeds_m.append(params[1:])
            labels_m.append(label_map[params[0]])           
        
embeds_m = np.array(embeds_m)
labels_m = np.array(labels_m)
embeds_a = np.array(embeds_a)
labels_a = np.array(labels_a)

print('embeds_m:', embeds_m.shape)
print('embeds_a:', embeds_a.shape)

# 
embeds = embeds_m
labels = labels_m
cluster_num = 20

kmeans = KMeans(n_clusters=cluster_num, random_state=0)
clusters = kmeans.fit_predict(embeds)

labels_pred = np.zeros_like(clusters)
for i in range(cluster_num):
    mask = (clusters == i)
    labels_pred[mask] = mode(labels[mask])[0]
    
from sklearn import metrics
mat = metrics.confusion_matrix(labels, labels_pred)
acc = metrics.accuracy_score(labels, labels_pred)
precision = metrics.precision_score(labels, labels_pred, average='macro')
f1_score = metrics.f1_score(labels, labels_pred, average='macro')
com_score = metrics.completeness_score(labels, labels_pred)
NMI = metrics.normalized_mutual_info_score(labels, labels_pred)
ARI = metrics.adjusted_rand_score(labels, labels_pred)

print(mat)
print('acc=', acc)
print('precision=', precision)
print('f1_score=', f1_score)
print('NMI=', NMI)
print('homo_score=', com_score)
print('ARI=', ARI)  


print('------------------------------------- for API clustering') 

embeds = embeds_a
labels = labels_a
cluster_num = 20


kmeans = KMeans(n_clusters=cluster_num, random_state=0)
clusters = kmeans.fit_predict(embeds)

labels_pred = np.zeros_like(clusters)
for i in range(cluster_num):
    mask = (clusters == i)
    labels_pred[mask] = mode(labels[mask])[0]
    
mat = metrics.confusion_matrix(labels, labels_pred)
acc = metrics.accuracy_score(labels, labels_pred)
precision = metrics.precision_score(labels, labels_pred, average='macro')
f1_score = metrics.f1_score(labels, labels_pred, average='macro')
com_score = metrics.completeness_score(labels, labels_pred)
NMI = metrics.normalized_mutual_info_score(labels, labels_pred)
ARI = metrics.adjusted_rand_score(labels, labels_pred)

print(mat)
print('acc=', acc)
print('precision=', precision)
print('f1_score=', f1_score)
print('NMI=', NMI)
print('homo_score=', com_score)
print('ARI=', ARI)  
