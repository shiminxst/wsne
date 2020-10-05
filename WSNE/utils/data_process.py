import csv
from utils import configs as config
import numpy as np
import random
import pickle as pkl
import os
import math
import scipy.sparse as sp
import sys
from scipy.sparse import lil_matrix
import tensorflow as tf
import copy


# build datasets from citeseer, cora and wiki
def build_sparse_nets(dataset, train_ratio, expand):
    """
        Args:
            dataset: name of dataset, i.e., wiki
            train_ratio: ratio of labeled nodes among all for training 
    """
    print("loading networks...")
    adjlist = {}
    nodenode_file = os.path.join(dataset, 'adjlist'+expand+'.txt')
    nodelabel_file = os.path.join(dataset, 'labels'+expand+'.txt')
    voca_file = os.path.join(dataset, 'vocabulary'+expand+'.txt')
    feature_file = os.path.join(dataset, 'content_token'+expand+'.txt')
    apimashup_ids = os.path.join(dataset, 'apimashup_ids'+expand+'.txt')
    
    top20category = os.path.join(dataset, 'top20category.txt')
    cats20 = []
    with open(top20category, 'r') as tr:
        for line in tr:
            ps = line.split()    
            cats20.append(ps[0])

    categoryNames_file = os.path.join(dataset, 'categoryNames'+expand+'.txt')
    cats = {}
    with open(categoryNames_file, 'r') as cr:
        for line in cr:
            ps = line.split()
            cats[ps[0]] = ps[1]

    api_ids = []
    with open(apimashup_ids, encoding='utf-8') as amfile:
        for line in amfile:
            ps = line.split()
            if ps[1] == 'A':
                api_ids.append(int(ps[0]))
    
    with open(nodenode_file, encoding='utf-8') as nnfile, open(nodelabel_file, encoding="utf-8") as nlfile:
        for line in nnfile:
            params = line.replace('\n', '').split()
            root = int(params[0])
            neighbors = [int(v) for v in params[1:]]
            adjlist[root] = neighbors
        node_num = len(adjlist.keys())
        print("node_num",node_num)
        
        print("------------train index generation begin...-------")
        node_labels = {}
        label_per_class = {}   
        candidate_ids = []
        for line in nlfile:
            params = line.replace('\n', '').split()
            node = int(params[0])
            
            if cats[params[1]] in cats20:
                candidate_ids.append(node)
            
            label = int(params[1].replace('L',''))
            
            if not label in label_per_class.keys():
                label_per_class[label] = [node]
            else:
                label_per_class[label].append(node)
            
            node_labels[node] = label
            
        label_num = len(label_per_class)
        print("label_num",label_num)
        print("actual samples per classes:",[(k, len(label_per_class[k])) for k in label_per_class.keys()])
        
#         train_indexes = []
#         for k in label_per_class.keys():
#             k_nodes = label_per_class[k]
#             indx = random.sample(k_nodes, train_ratio)
#             train_indexes = train_indexes + indx
            
        print("candidate train num:", len(candidate_ids))
        total_train_num = int(train_ratio * len(candidate_ids))
#         total_train_num = int(train_ratio * node_num)
        train_indexes = random.sample(candidate_ids, total_train_num)
#         train_indexes = random.sample(range(node_num), total_train_num)
        train_nodes = sorted([(idx, node_labels[idx]) for idx in train_indexes], key=lambda x: x)
        print(train_nodes)

        train_label_per_class = {}     
        with open(os.path.join(dataset,str(train_ratio)+"_train_index.txt"), 'w') as tiw:
            for chunk in train_nodes:
                if not chunk[1] in train_label_per_class.keys():
                    train_label_per_class[chunk[1]] = 1
                else:
                    train_label_per_class[chunk[1]] = train_label_per_class[chunk[1]] + 1
                ti_text = str(chunk[0]) + ' ' + str(chunk[1])
                tiw.write(ti_text+'\n') 
          
        print("training samples per classes:{}".format(train_label_per_class))
        print("------------train index generation end-------")
        tiw.close()
        
    nnfile.close()
    nlfile.close()
    
    adjlist = [(k,adjlist[k]) for k in sorted(adjlist.keys())]
    nnAdjM = lil_matrix((node_num, node_num))    
    for chunk in adjlist:
        root_node = chunk[0]
        adj_node = chunk[1]
        nnAdjM[root_node, adj_node] = 1    
    
    nnAdjM = nnAdjM + np.identity(node_num)
    d_nnl_diag = np.squeeze(np.sum(np.array(nnAdjM), axis=1))
    d_nnl_inv_sqrt_diag = np.power(d_nnl_diag, -1/2)
    d_nnl_inv_sqrt = np.diag(d_nnl_inv_sqrt_diag)
    nnAdj_norm = np.dot(np.dot(d_nnl_inv_sqrt, nnAdjM), 
                          d_nnl_inv_sqrt)
    
    nnAdj_norm_mashup = copy.deepcopy(nnAdj_norm)
    for aid in api_ids:
        nnAdj_norm_mashup[aid]=0
                          
    nn_embeddings_identity = np.identity(n=node_num)
                          
#     nn_embeddings_feature = []
#     with open(feature_file, encoding="utf-8") as fefile:
#         for line in fefile:
#             params = line.replace('\n', '').split()
#             node_id = params[0]
#             feats = [float(v) for v in params[1:]]
#             for i in range(len(feats)):
#                 if feats[i] != 0:
#                     feats[i] = 1.
#             nn_embeddings_feature.append(feats)
#     nn_embeddings_feature = np.array(nn_embeddings_feature)
    voca_dic = {}
    node_features = {}
    with open(voca_file, encoding='utf-8') as vofile, open(feature_file, encoding="utf-8") as fefile:
        for line in vofile:
            params = line.replace('\n', '').split('=')
            voca_dic[params[1]] = int(params[0])
        
        feature_num = len(voca_dic.keys())
        print('Total number of features:', feature_num)
        
        li = 0
        for line in fefile:
            params = line.replace('\n', '').split()
            word_list = []
            for word in params:
                word_list.append(voca_dic[word])
            node_features[li] = word_list
            li += 1
    nn_embeddings_feature = lil_matrix((node_num, feature_num))
    for i in range(node_num):
        feats = node_features[i]
        nn_embeddings_feature[i, feats] = 1
    
    print('nn_embeddings_feature_shape:',nn_embeddings_feature.shape)
        
    node_labels = [(k,node_labels[k]) for k in sorted(node_labels.keys())]    
    node_class = np.zeros([node_num,label_num])
    for chunk in node_labels:
        root_node = chunk[0]
        labels = chunk[1]
        node_class[root_node, labels] = 1
    
    csr_adj_nn = sp.csr_matrix(nnAdjM)
    csr_adj_nn_norm = sp.csr_matrix(nnAdj_norm)
    csr_adj_nn_norm_mashup = sp.csr_matrix(nnAdj_norm_mashup)
    csr_node_class = sp.csr_matrix(node_class)
    csr_nn_embed_identity = sp.csr_matrix(nn_embeddings_identity)    
    csr_nn_embed_feature = sp.csr_matrix(nn_embeddings_feature)    
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m1.adj'), 'wb')
    pkl.dump(csr_adj_nn, f)
    f.close()  
    np.savetxt('out.nn_adj',csr_adj_nn.toarray()) 
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m_norm1.adj'), 'wb')
    pkl.dump(csr_adj_nn_norm, f)
    f.close()   
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m_norm_mashup1.adj'), 'wb')
    pkl.dump(csr_adj_nn_norm_mashup, f)
    f.close()   
    
    f = open(os.path.join(dataset, str(train_ratio)+'_m1.label'), 'wb')
    pkl.dump(csr_node_class, f)
    f.close()   
    np.savetxt('out.label',csr_node_class.toarray())  
    
    f = open(os.path.join(dataset, str(train_ratio)+'_identity1.x'), 'wb')
    pkl.dump(csr_nn_embed_identity, f)
    f.close() 
           
    f = open(os.path.join(dataset, str(train_ratio)+'_feature1.x'), 'wb')
    pkl.dump(csr_nn_embed_feature, f)
    f.close()        
#     np.savetxt('out.feature',csr_nn_embed_feature.toarray()[0:20])  
    
# build_sparse_nets(config.FILES.API, 0.7, '0')      


def load_data(dataset, train_ratio, x_flag='feature', expand=''):
    """
    Loads input corpus from gcn/data directory

    m.x => the feature vectors of all nodes and labels as scipy.sparse.csr.csr_matrix object;
    m.adj => the adjacency matrix of node-node-label network as scipy.sparse.csr.csr_matrix object;
    m.label => the labels for all nodes as scipy.sparse.csr.csr_matrix objectt;
    train_index.txt => the indices of labeled nodes for supervised training as numpy.ndarray object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    train_ratio = str(train_ratio)
    names = [train_ratio+"_"+x_flag+'1.x', train_ratio+'_m1.adj', 
             train_ratio+'_m_norm1.adj', train_ratio+'_m_norm_mashup1.adj', train_ratio+'_m1.label']
    
    objects = []
    for i in range(len(names)):
        with open(os.path.join(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, adj, adj_norm, adj_norm_mashup, label = tuple(objects)
    train_indexes = []
    label_counts = {}
    balance_num = 0
    with open(os.path.join(dataset,train_ratio+"_train_index.txt"), 'r') as tir:
        for line in tir:
            params = line.split()
            train_indexes.append(int(params[0]))
            if not params[1] in label_counts.keys():
                label_counts[params[1]] = [int(params[0])]
            else:
                label_counts[params[1]].append(int(params[0]))
    
            if balance_num < len(label_counts[params[1]]):
                balance_num = len(label_counts[params[1]])
    
    # print the class distribution
#     label_dist = [(k,len(label_counts[k])) for k in sorted(label_counts.keys())]   
    label_dist = [(kv[0], len(kv[1])) for kv in sorted(label_counts.items(), key=lambda x: len(x[1]), reverse=True)]    
   
    print('label_distribution:', label_dist)
    print('balance_num:', balance_num)
#         id_text = tir.readline()
#         train_indexes = np.array([int(i) for i in id_text.split()])
    train_indexes = np.array(train_indexes)
    node_num = adj.shape[0]


    apimashup_ids = os.path.join(dataset, 'apimashup_ids'+expand+'.txt')
    apimashup_flags = [False]*node_num
    with open(apimashup_ids, encoding='utf-8') as amfile:
        for line in amfile:
            ps = line.split()
            if ps[1] == 'A':
                apimashup_flags[int(ps[0])]=True

    
    # construct test set from the selected 20 categories
    top20category = os.path.join(dataset, 'top20category.txt')
    cats20 = []
    with open(top20category, 'r') as tr:
        for line in tr:
            ps = line.split()    
            cats20.append(ps[0])
    
    categoryNames_file = os.path.join(dataset, 'categoryNames'+expand+'.txt')
    labels_file = os.path.join(dataset, 'labels'+expand+'.txt')
    cats = {}
    allnodeids = []
    unique_labs = []
    with open(categoryNames_file, 'r') as cr, open(labels_file, 'r') as lr:
        for line in cr:
            ps = line.split()
            cats[ps[0]] = ps[1]
        print(cats20)    
        for line in lr:
            ps = line.split()
            if cats[ps[1]] in cats20:
                allnodeids.append(int(ps[0]))
                if cats[ps[1]] not in unique_labs:
                    unique_labs.append(cats[ps[1]])
    print('unique_labs:', unique_labs)
    print('allnodeids:', len(allnodeids))
    
    
    temp_indexes = np.setdiff1d(np.array(allnodeids), train_indexes)                
    print('temp_indexes:', len(temp_indexes))
#     temp_indexes = np.setdiff1d(np.array(range(node_num)), train_indexes)                
            
    
    test_indexes = np.array(random.sample(list(temp_indexes), int(0.9 * len(temp_indexes))))
    
    print(x.shape, adj.shape, label.shape, train_indexes.shape, test_indexes.shape)
    
    return x, adj, adj_norm, adj_norm_mashup, label, train_indexes, test_indexes, apimashup_flags

# load_data(config.FILES.cora, str(0.1)) 



def process_servicedata(dataset, expand):
    
    # inputs
    node_file = os.path.join(dataset, 'adjlist_expand'+expand+'.txt')
    label_file = os.path.join(dataset, 'labels.txt')
    feature_file = os.path.join(dataset, 'content_token.txt')
    apimashup_ids_file = os.path.join(dataset, 'apimashup_ids.txt')
    categoryNames_file = os.path.join(dataset, 'categoryNames.txt')
    
    # outputs
    vocabulary = os.path.join(dataset, 'vocabulary'+expand+'.txt')
    adjlist = os.path.join(dataset, 'adjlist'+expand+'.txt')
    labels = os.path.join(dataset, 'labels'+expand+'.txt')
    content_token = os.path.join(dataset, 'content_token'+expand+'.txt')
    apimashup_ids = os.path.join(dataset, 'apimashup_ids'+expand+'.txt')
    categoryNames = os.path.join(dataset, 'categoryNames'+expand+'.txt')
    
    idmap = {}
    adjcollection = []
    with open(node_file, 'r') as nf:
        for line in nf:
            ps = line.replace('\n','').split(' ')
            
            idlist = []
            for p in ps:
                if not p in idmap.keys():
                    idmap[p] = len(idmap)
                idlist.append(idmap[p])
            adjcollection.append(idlist)
    
    labdic = {}
    labmap = {}
    with open(label_file, 'r') as lf:
        for line in lf:
            ps = line.replace('\n','').split(' ')
            
            if ps[0] in idmap.keys():
                
                if not ps[1] in labmap.keys():
                    labmap[ps[1]] = 'L'+str(len(labmap))
                id_new = idmap[ps[0]]
                labdic[id_new] = labmap[ps[1]]
                
    idenmap = {}
    with open(apimashup_ids_file, 'r') as af:
        for line in af:
            ps = line.split()
            if ps[0] in idmap.keys():
                idenmap[idmap[ps[0]]] = ps[1]
    
    catenamemap = {}
    with open(categoryNames_file, 'r') as cf:
        for line in cf:
            ps = line.split()
            ps_1 = ps[1]
            if len(ps)>2:
                ps_1 = '_'.join(ps[1:])
            if ps[0] in labmap.keys():
                catenamemap[labmap[ps[0]]] = ps_1
    
    index = 0
    contdict = {}
    vocadict = {}
    with open(feature_file, 'r') as ff:
        for line in ff:
            if str(index) in idmap.keys():
                id_new = idmap[str(index)]
                contdict[id_new] = line
                ps = line.split()
                for p in ps:
                    if not p in vocadict.keys():
                        vocadict[p] = len(vocadict)
            index += 1   
    
    contdict = [(k,contdict[k]) for k in sorted(contdict.keys())]
    vocadict = sorted(vocadict.items(), key=lambda x: x[1])
    
    with open(adjlist, 'w') as adjw, open(labels, 'w') as labw, open(content_token, 'w') as contw, open(vocabulary, 'w') as vocw, open(apimashup_ids, 'w') as aw, open(categoryNames, 'w') as cw:
         
        for elem in adjcollection:
            line = ' '.join([str(e) for e in elem])
            adjw.write(line+'\n')
        for k in labdic.keys():
            line = str(k) + ' ' + labdic[k]
            labw.write(line+'\n')
        for k,v in contdict:
            contw.write(v)
        for kv in vocadict:
            line = str(kv[1]) + '=' + kv[0]
            vocw.write(line+'\n')
        for k in idenmap.keys():
            line = str(k) + ' ' + idenmap[k]
            aw.write(line+'\n')
        for k in catenamemap.keys():
            line = str(k) + ' ' + catenamemap[k]
            cw.write(line+'\n')
            
    
# process_servicedata(config.FILES.API, '2')


def category_statisitcs(dataset):
    
    labels = os.path.join(dataset, 'labels.txt')
    categoryNames = os.path.join(dataset, 'categoryNames.txt')
    top20category = os.path.join(dataset, 'top20category.txt')
    
    cats = {}
    with open(categoryNames, 'r') as cr:
        for line in cr:
            ps = line.split()
            cats[ps[0]] = ps[1]
    
    label_counts = {}
    with open(labels, 'r') as lr:
        for line in lr:
            params = line.split()
            
            if not cats[params[1]] in label_counts.keys():
                label_counts[cats[params[1]]] = [int(params[0])]
            else:
                label_counts[cats[params[1]]].append(int(params[0]))            

    label_dist = [(kv[0], len(kv[1])) for kv in sorted(label_counts.items(), key=lambda x: len(x[1]), reverse=True)]
    print('label_distribution:', label_dist)
    
    
    labels = os.path.join(dataset, 'labels0.txt')
    categoryNames = os.path.join(dataset, 'categoryNames0.txt')
    
    cats = {}
    with open(categoryNames, 'r') as cr:
        for line in cr:
            ps = line.split()
            cats[ps[0]] = ps[1]
    
    label_counts = {}
    with open(labels, 'r') as lr:
        for line in lr:
            params = line.split()
            
            if not cats[params[1]] in label_counts.keys():
                label_counts[cats[params[1]]] = [int(params[0])]
            else:
                label_counts[cats[params[1]]].append(int(params[0]))            

    label_dist = [(kv[0], len(kv[1])) for kv in sorted(label_counts.items(), key=lambda x: len(x[1]), reverse=True)]
    
    with open(top20category, 'w') as tw:
        for kv in label_dist[0:20]:
            line = kv[0] + ' ' + str(kv[1])
            tw.write(line+'\n')
        
    
    print('label_distribution:', label_dist)
    
    labels = os.path.join(dataset, 'labels2.txt')
    categoryNames = os.path.join(dataset, 'categoryNames2.txt')
    
    cats = {}
    with open(categoryNames, 'r') as cr:
        for line in cr:
            ps = line.split()
            cats[ps[0]] = ps[1]
    
    label_counts = {}
    with open(labels, 'r') as lr:
        for line in lr:
            params = line.split()
            
            if not cats[params[1]] in label_counts.keys():
                label_counts[cats[params[1]]] = [int(params[0])]
            else:
                label_counts[cats[params[1]]].append(int(params[0]))            

    label_dist = [(kv[0], len(kv[1])) for kv in sorted(label_counts.items(), key=lambda x: len(x[1]), reverse=True)]
    print('label_distribution:', label_dist)

# category_statisitcs(config.FILES.API)

def trainset_statistics(dataset):
    
    labels = os.path.join(dataset, '0.1_train_index.txt')
    categoryNames = os.path.join(dataset, 'categoryNames0.txt')
    
    cats = {}
    with open(categoryNames, 'r') as cr:
        for line in cr:
            ps = line.split()
            cats[ps[0]] = ps[1]    
 
    label_counts = {}
    with open(labels, 'r') as lr:
        for line in lr:
            params = line.split()
            
            if not cats['L'+params[1]] in label_counts.keys():
                label_counts[cats['L'+params[1]]] = [int(params[0])]
            else:
                label_counts[cats['L'+params[1]]].append(int(params[0]))            

    label_dist = [(kv[0], len(kv[1])) for kv in sorted(label_counts.items(), key=lambda x: len(x[1]), reverse=True)]
    print('label_distribution:', label_dist)
    
# trainset_statistics(config.FILES.API)

    
    
    
    







