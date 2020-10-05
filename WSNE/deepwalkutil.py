# -*- coding: utf-8 -*-
from _overlapped import NULL

__author__ = 'Min Shi'
__date__ = '07/29/2018'

import collections
import os
import gensim
import random
import gensim.utils as ut
import URLs as URL
import numpy as np
import gc
from collections import namedtuple
from itertools import compress
from numpy import dtype
from deepwalk import graph


WordTopic = namedtuple('WordTopic', 'words topics docId')
NodeTopic = namedtuple('NodeTopic', 'nodes topics')

def loadNetworkContent(dir, T, stemmer=0, topicN=1):
    alldocs = []
    contentfile = os.path.join(dir,"content_token.txt")
    labelfile = os.path.join(dir,'topics'+str(topicN)+'_T_'+ str(T)+'.txt')
    
    doc_labels = dict()
    with open(labelfile) as lr:
        for lline in lr:
            params = lline.split()
            doc_labels[int(params[0])] = params[1].split(",")
    
    with open(contentfile, 'r', encoding="iso-8859-1") as cr:
        linecount = 0
        for cline in cr:
            if stemmer == 1:
                cline = gensim.parsing.stem_text(cline)
            else:
                cline = cline.lower()
            words = ut.to_unicode(cline).split()
            topics = []
            if linecount in doc_labels.keys():
                topics= doc_labels[linecount]
            alldocs.append(WordTopic(words, topics, str(linecount)))
            linecount += 1
    return alldocs

# alldocs = loadNetworkContent(URL.FILES.serviceData)   
# print(alldocs[0]) 

def generateWalks(dir, number_walks=20, walk_length=10):
    adjlistfile = os.path.join(dir,"adjlist.txt")
    print('number_walks:',number_walks)
    G = graph.load_adjacencylist(adjlistfile)
    num_walks = len(G.nodes()) * number_walks
    print("Number of nodes:", len(G.nodes()))
    print("Number of walks:", num_walks)
    
    print("walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths= number_walks, 
                                        path_length=walk_length, alpha=0, rand=random.Random(1))
    
    topical_raw_walks = []
    walkfile = os.path.join(dir,"walks.txt")
    with open(walkfile, 'w') as ww:
        for i, x in enumerate(walks):
            nodes = [str(t) for t in x]
            topical_raw_walks.append(nodes)
            ww.write(' '.join(nodes)+"\n")
    return topical_raw_walks
    
# def main():
#     generateWalks(URL.FILES.serviceData, 50, 10)
# if __name__ == '__main__':
#     main()

def build_dictionary(raw_data):
    
    item_dictionary = dict()
    topic_dictionary = dict()
    all_items = []
    for items in raw_data:
        all_items.extend(items)
    item_count = []
    item_count.extend(collections.Counter(all_items).most_common())
    for item,_ in item_count:
        item_dictionary[item] = len(item_dictionary)
    
    return item_dictionary  

def generate_batch_deepwalk(dir, node_node_batch_size, 
                            num_skip, node_window_size):
    print("Generating the training instances for DeepWalk models...")
    allnodes = generateWalks(dir)
    node_dict = build_dictionary(allnodes)
    print('node_dict:',len(node_dict))
    reverse_node_dict = dict(zip(node_dict.values(), node_dict.keys()))
    
    skip_window = int(node_window_size // 2)
    assert num_skip <= 2 * skip_window
    
    nn_all_instances = []
    nn_all_labels = []
    count = 0
    
    for nodes in allnodes:
        
        for center_index in range(node_window_size, len(nodes)-(node_window_size+1)):
            buffer = []
            
            data_index = center_index
            for _ in range(skip_window):
                data_index = data_index - 1
                if data_index >= 0:
                    buffer.append(data_index)
                    
            data_index = center_index
            for _ in range(skip_window):
                data_index = data_index + 1
                if data_index <= (len(nodes) - 1):
                    buffer.append(data_index);
            
            object_to_avoid = []
            object_index = random.randint(0, len(buffer) - 1)
            for j in range(num_skip):
                if j >= len(buffer):
                    break
                while object_index in object_to_avoid:
                    object_index = random.randint(0, len(buffer) - 1)
                
                object_to_avoid.append(object_index)
               
                # node-node training instance
                nn_all_instances.append(node_dict[nodes[center_index]])
                nn_all_labels.append(node_dict[nodes[buffer[object_index]]])
                
        count += 1
        if count % 5000 == 0:
            gc.collect()
            
    print('nn_all_instances samples: ', nn_all_instances[0])
    print('nn_all_labels samples: ', nn_all_labels[0])
    
    nn_batchs_num = len(nn_all_instances) // node_node_batch_size
    if len(nn_all_instances) % node_node_batch_size > 0:
        nn_batchs_num += 1
        nn_residual_num = node_node_batch_size - len(nn_all_instances) % node_node_batch_size
        print("nn_residual_num: ", nn_residual_num)
        nn_all_instances.extend(nn_all_instances[:nn_residual_num])   
        nn_all_labels.extend(nn_all_labels[:nn_residual_num]) 
    print("nn_batchs_num: ", nn_batchs_num)
    
    nn_all_instances = np.array(nn_all_instances)
    nn_all_labels = np.array(nn_all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(nn_all_instances))) 
    nn_all_instances = nn_all_instances[shuffle_indices]
    nn_all_labels = nn_all_labels[shuffle_indices]
    
    nn_training_batches = [(nn_all_instances[i*node_node_batch_size:(i+1)*node_node_batch_size], 
                         nn_all_labels[i*node_node_batch_size:(i+1)*node_node_batch_size]) for i in range(nn_batchs_num)]
    return nn_training_batches, reverse_node_dict

# def main():
#     nn_training_batches, reverse_node_dict = generate_batch_deepwalk(URL.FILES.coraData, 120, 2, 2)              
#     print(len(reverse_node_dict))
# #     print(reverse_node_dict[10309])
# if __name__ == '__main__':
#     main()    



def repository_size(dir, T, topicN = 1):
    content_token = os.path.join(dir, "content_token.txt")
#     labels = os.path.join(dir, 'topics'+str(topicN)+'_T_'+ str(T)+'.txt')
    adjlist = os.path.join(dir, "adjlist.txt")
    
    word_idct = dict()
    node_idct = dict()
    with open(content_token, 'r', encoding="iso-8859-1") as cr, open(adjlist) as ar:
        
        for l1 in cr:
            for w in l1.split():
                word_idct[w] = 1
        for l3 in ar:
            for n in l3.split():
                node_idct[n] = 1
    return len(node_idct), len(word_idct)

# node_size, topic_size, word_size = repository_size(URL.FILES.coraData)  
# print('node_size:{}, topic_size:{}, word_size:{}'.format(node_size,topic_size,word_size))  

def convertDataFormat_citeseer(dir):
    
    docfile = os.path.join(dir, "citeseer.features")
    adjedgesfile = os.path.join(dir, "citeseer_edgelist.txt")
    labelsfile = os.path.join(dir, "groups.txt")
    
    content_token = os.path.join(dir, "content_token.txt")
    labels = os.path.join(dir, "labels.txt")
    adjlist = os.path.join(dir, "adjlist.txt")
    
    docid_dict = dict()
    with open(docfile, 'r') as dr, open(content_token, 'w') as cw:
        for line in dr:
            params = line.split()
            
            words = ['w'+str(index) for index, value in enumerate(params[1:]) if float(value) != 0]
#             if len(words) == 0:
#                 continue
            docid_dict[params[0]] = len(docid_dict)
            
            cw.write(' '.join(words)+"\n")
            
    label_dict = dict()
    with open(labelsfile, 'r') as lr, open(labels, 'w') as lw:
         
        for line in lr:
            params = line.split()
#             if not params[0] in docid_dict.keys():
#                 continue
            nodeid = docid_dict[params[0]]
            if not params[1] in label_dict.keys():
                label_dict[params[1]] = 'L' + str(len(label_dict))
            lw.write(str(nodeid) + ' ' + label_dict[params[1]] + '\n')
            
    adjnodes = dict()
    with open(adjedgesfile, 'r') as gr:
        for line in gr:
            ns = line.split()
            
            if not ns[0] in docid_dict.keys() or not ns[1] in docid_dict.keys():
                continue
            
            ns[0] = str(docid_dict[ns[0]])
            ns[1] = str(docid_dict[ns[1]])
            
            if not ns[0] in adjnodes.keys():
                adjnodes[ns[0]] = [ns[1]]
            elif not ns[1] in adjnodes[ns[0]]:
                adjnodes[ns[0]].append(ns[1])
            
            if not ns[1] in adjnodes.keys():
                adjnodes[ns[1]] = [ns[0]]
            elif not ns[0] in adjnodes[ns[1]]:
                adjnodes[ns[1]].append(ns[0])
    with open(adjlist, 'w') as aw:
        
        for k in adjnodes.keys():
            aw.write(k + ' ' + ' '.join(adjnodes[k]) + '\n')
            
# convertDataFormat_citeseer(URL.FILES.citeseerData)     
            
def convertDataFormat_wiki(dir):
    
    docfile = os.path.join(dir, "citeseer.features")
    adjedgesfile = os.path.join(dir, "citeseer_edgelist.txt")
    labelsfile = os.path.join(dir, "groups.txt")
    
    content_token = os.path.join(dir, "rtm_content_token.txt")
    labels = os.path.join(dir, "rtm_labels.txt")
    adjlist = os.path.join(dir, "rtm_adjlist.txt")
    
    docid_dict = dict()
    with open(docfile, 'r') as dr, open(content_token, 'w') as cw:
        for line in dr:
            params = line.split()
            
            words = ['w'+str(index) for index, value in enumerate(params[1:]) if float(value) != 0]
            if len(words) == 0:
                continue
            docid_dict[params[0]] = len(docid_dict)
            
            cw.write(' '.join(words)+"\n")
            
    label_dict = dict()
    with open(labelsfile, 'r') as lr, open(labels, 'w') as lw:
         
        for line in lr:
            params = line.split()
            if not params[0] in docid_dict.keys():
                continue
            nodeid = docid_dict[params[0]]
            if not params[1] in label_dict.keys():
                label_dict[params[1]] = 'L' + str(len(label_dict))
            lw.write(str(nodeid) + ' ' + label_dict[params[1]] + '\n')
            
    adjnodes = dict()
    with open(adjedgesfile, 'r') as gr:
        for line in gr:
            ns = line.split()
            
            if not ns[0] in docid_dict.keys() or not ns[1] in docid_dict.keys():
                continue
            
            ns[0] = str(docid_dict[ns[0]])
            ns[1] = str(docid_dict[ns[1]])
            
            if not ns[0] in adjnodes.keys():
                adjnodes[ns[0]] = [ns[1]]
            elif not ns[1] in adjnodes[ns[0]]:
                adjnodes[ns[0]].append(ns[1])
            
            if not ns[1] in adjnodes.keys():
                adjnodes[ns[1]] = [ns[0]]
            elif not ns[0] in adjnodes[ns[1]]:
                adjnodes[ns[1]].append(ns[0])
    with open(adjlist, 'w') as aw:
        
        for k in adjnodes.keys():
            aw.write(k + ' ' + ' '.join(adjnodes[k]) + '\n')
   
# convertDataFormat_wiki(URL.FILES.citeseerData)    