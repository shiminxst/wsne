# -*- coding: utf-8 -*-

# Network Data pre-prcesssing

__author__ = 'Min Shi'
__date__ = '26-July-2018'

import URLs as URL
import os
import csv
import gensim
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')
# 
# stop_words = set(stopwords.words('english'))

def buildServiceNet(originalAPI, apiFile, mashupFile):
    # output files
#     cntFile = os.path.join(URL.FILES.serviceData,"content_token.txt")
#     graphFile = os.path.join(URL.FILES.serviceData, "adjlist.txt")
#     labelFile = os.path.join(URL.FILES.serviceData, "labels.txt")
#     labelnameFile = os.path.join(URL.FILES.serviceData, "labelNames.txt")
    cntFile = os.path.join(URL.FILES.serviceData,"content_token.txt")
    graphFile = os.path.join(URL.FILES.serviceData, "adjlist.txt")
    labelFile = os.path.join(URL.FILES.serviceData, "labels.txt")
    metadataFile = os.path.join(URL.FILES.serviceData, "functional_metadata.txt")
    labelnameFile = os.path.join(URL.FILES.serviceData, "categoryNames.txt")
    serviceflagFile = os.path.join(URL.FILES.serviceData, "apimashup_ids.txt")
    
    contents = []
    labels = dict()
    metadata = dict()
    links = dict()
    
    labelist = []
    apinamelist = []
    
    service_flag = dict()
    
#     validapis = []
#     with open(mashupFile, newline='', encoding='iso-8859-1') as mashupcsv:
#         spamreader = csv.reader(mashupcsv, delimiter=',')
#         header = next(spamreader, None)
#         for row in spamreader:
#             mashup_apis = str.lower(row[3]).split(",")
#             for apin in mashup_apis:
#                 apin = str.strip(apin)
#                 if not apin in validapis:
#                     validapis.append(apin)
#     print("validapis:",len(validapis))

    with open(originalAPI, newline='', encoding='iso-8859-1') as originalapicsv:
        spamreader = csv.reader(originalapicsv, delimiter=',')
        header = next(spamreader, None)
        for row in spamreader:
            api_name = str.lower(row[0])
            api_tags = str.lower(row[2]).split("###")
            api_content = str.lower(row[3])
#             api_content.replace('\n',' ')
            api_content = api_content.replace('\n', ' ').replace('\r', '')
            
            apinamelist.append(api_name)
            
            contents.append(api_content)
            
            labs = []
            if not api_tags[0] in labelist:
                labelist.append(api_tags[0])
            labs.append("L"+str(labelist.index(api_tags[0])))
            labels[len(contents)-1] = labs
            metadata[len(contents)-1] = api_tags
            service_flag[len(contents)-1] = 'A'
            
    
    with open(apiFile, newline='', encoding='iso-8859-1') as apicsv:
        spamreader = csv.reader(apicsv, delimiter=',')
        header = next(spamreader, None)
        for row in spamreader:
            api_name = str.lower(row[0])
#             if not api_name in validapis:
#                 print(api_name)
#                 continue
            
            if api_name in apinamelist:
                continue
            
            api_tags = str.lower(row[1]).split("###")
            api_content = str.lower(row[2])
#             api_content.replace('\n',' ')
            api_content = api_content.replace('\n', ' ').replace('\r', '')
            
            apinamelist.append(api_name)
            
            contents.append(api_content)
            
            labs = []
            if not api_tags[0] in labelist:
                labelist.append(api_tags[0])
#             for tag in api_tags:
#                 if not tag in labelist:
#                     labelist.append(tag)
#                 if not "L"+str(labelist.index(tag)) in labs:
#                     labs.append("L"+str(labelist.index(tag)))
            labs.append("L"+str(labelist.index(api_tags[0])))
            labels[len(contents)-1] = labs
            metadata[len(contents)-1] = api_tags
            service_flag[len(contents)-1] = 'A'
            
    print("Total number of labels:", len(labelist))
    print("contents:", len(contents))
    print("links:", len(links))
    print("labels:", len(labels))
    print("\n")
    linkcount = 0
    with open(mashupFile, newline='', encoding='iso-8859-1') as mashupcsv:
        spamreader = csv.reader(mashupcsv, delimiter=',')
        header = next(spamreader, None)
        for row in spamreader:
            mashup_name = str.lower(row[0])
            mashup_tags = str.lower(row[1]).split("###")
            mashup_content = str.lower(row[2])
            mashup_apis = str.lower(row[3]).split(",")
            
#             cline = gensim.parsing.stem_text(mashup_content)
#             cline = word_tokenize(cline)
#             cline = [w for w in cline if not w in stop_words]
            contents.append(mashup_content)
            
            labs = []
#             for tag in mashup_tags:
#                 if not tag in labelist:
#                     labelist.append(tag)
            if not mashup_tags[0] in labelist:
                labelist.append(mashup_tags[0])
#                 if not "L"+str(labelist.index(tag)) in labs:
#                     labs.append("L"+str(labelist.index(tag)))
            labs.append("L"+str(labelist.index(mashup_tags[0])))
            labels[len(contents)-1] = labs
            metadata[len(contents)-1] = mashup_tags
            service_flag[len(contents)-1] = 'M'
            
            apils = []
#             mashupindex = 'm_'+str(len(contents)-1)
            mashupindex = str(len(contents)-1)
            for apin in mashup_apis:
                apin = str.strip(apin)
                if apin in apinamelist:
#                     apiindex = 'a_'+str(apinamelist.index(apin))
                    apiindex = str(apinamelist.index(apin))
                    if not apiindex in links.keys():
                        links[apiindex] = []
                    links[apiindex].append(str(mashupindex))
                    apils.append(str(apiindex))
                    linkcount += 1
            links[mashupindex] = apils
            
                
#     with open(labelnameFile, 'w') as lnw:
#         for lname in labelist:
#             lnw.write(lname+"\n")
            
    print("Total number of labels:", len(labelist))
    print("contents:", len(contents))
    print("links:", len(links))
    print("labels:", len(labels))
    print("linkcount:", linkcount)
    
    with open(cntFile, 'w',encoding='iso-8859-1') as cw:
        for i in  range(len(contents)):
            cw.write(contents[i]+"\n")
            
    with open(labelFile, 'w',encoding='iso-8859-1') as lw:
        for key in labels:
            lw.write(str(key) +" " + ','.join(labels[key]) + "\n")
            
    with open(metadataFile, 'w',encoding='iso-8859-1') as mw:
        for key in metadata:
            mw.write(' '.join(metadata[key]) + "\n")
            
    with open(serviceflagFile, 'w',encoding='iso-8859-1') as sw:
        for key in service_flag:
            sw.write(str(key) +" " + service_flag[key] + "\n")
            
    with open(graphFile, 'w',encoding='iso-8859-1') as gw:
        for key in links:
            gw.write(str(key) +" " + ' '.join(links[key]) + "\n")
            
    with open(labelnameFile, 'w',encoding='iso-8859-1') as lnw:
        for i in range(len(labelist)):
            lnw.write("L" + str(i) + ' ' + labelist[i] + "\n")  
             
# buildServiceNet(os.path.join(URL.FILES.serviceData,"APIs.csv"), os.path.join(URL.FILES.serviceData,"raw_APIs.csv"), os.path.join(URL.FILES.serviceData,"raw_Mashups.csv"))


def label_statistic(data):
    cntFile = os.path.join(data,"all_descs.txt")
    graphFile = os.path.join(data, "composition.txt")
    labelFile = os.path.join(data, "category.txt")
    
    cntFile_freq = os.path.join(data,"content_token.txt")
    graphFile_freq = os.path.join(data, "adjlist.txt")
    labelFile_freq = os.path.join(data, "labels.txt")
    
    labels = dict()
    selected_labels = []
    with open(labelFile) as lr:
        for line in lr:
            params = line.split()
            if not params[1] in labels.keys():
                labels[params[1]] = [params[0]]
            else:
                labels[params[1]].append(params[0])
    label_dict = dict()
    for key, value in sorted(labels.items(), key=lambda x: len(x[1]), reverse=True):
        if len(value) >= 50:
            if not key in label_dict.keys():
                label_dict[key] = 'L'+str(len(label_dict))
            selected_labels.append(key)
    print('selected_labels:',len(selected_labels))
    
    node_ids = dict()
    node_ids_reverse = dict()
    with open(labelFile) as lr, open(labelFile_freq, 'w') as lw:
        for line in lr:
            params = line.split()
            if params[1] in selected_labels:
                node_ids[params[0]] = len(node_ids)
                node_ids_reverse[len(node_ids)] = int(params[0])
                lw.write(str(node_ids[params[0]]) + ' ' + label_dict[params[1]] + '\n')
    print(node_ids_reverse)
    conts = dict()
    with open(cntFile, 'r', encoding='iso-8859-1') as cr, open(cntFile_freq, 'w', encoding='iso-8859-1') as cw:
        
        lcount = 0
        for line in cr:
            conts[lcount] = line;
            lcount += 1
        
        for k in node_ids_reverse.keys():
            cw.write(conts[node_ids_reverse[k]])
            
    with open(graphFile) as gr, open(graphFile_freq, 'w') as gw:
        
        for line in gr:
            id = line.split()[0]
            linked_ids = line.split()[1:]
            
            if id in node_ids.keys():
                id_new = str(node_ids[id])
                
                linked_ids_new = ' '.join([str(node_ids[value]) for value in linked_ids if value in node_ids.keys()])
                
                gw.write(id_new + ' ' + linked_ids_new + '\n')
            
# label_statistic(URL.FILES.serviceData)  


# invoking frequency - original data
def apiFreq(data):
    freq = dict()
    
    count  = 0
    graphFile = os.path.join(data, "adjlist.txt")
    with open(graphFile) as gr:
        for line in gr:
            if line[0] == 'm':
                continue
            count += 1
            used_time = len(line.split()) - 1
            if not used_time in freq.keys():
                freq[used_time] = 1
            else:
                freq[used_time] += 1
    
    print(count)
    print(freq)
    
# apiFreq(URL.FILES.serviceData)    
    
    
def statistics(data):
        
    cntFile = os.path.join(data,"content_token.txt")
    adjlistFile = os.path.join(data, "adjlist.txt")
    labelFile = os.path.join(data, "labels.txt")
    metadataFile = os.path.join(data, "functional_metadata.txt")
    
    linecount = 0
    totalwords = 0
    vocabulary = []
    with open(cntFile) as cr:
        for line in cr:
            for w in line.split():
                if not w in vocabulary:
                    vocabulary.append(w)
            totalwords += len(line.split())
            linecount += 1
    print('Vocabulary size:', len(vocabulary))
    print('Average number of words per node:',totalwords/linecount)
    
    linkcount = 0
    with open(adjlistFile) as ar:
        for line in ar:
            linkcount += len(line.split())-1
    print('Total number of links:', linkcount)
    
    labels = []
    with open(labelFile) as lr:
        for line in lr:
            label = line.split()[1] 
            if not label in labels:
                labels.append(label)
    print('Total number of labels:', len(labels))
    
    tags = []
    tagcounts = 0
    with open(metadataFile) as mr:
        for line in mr:
            for t in line.split():
                if not t in tags:
                    tags.append(t)
            tagcounts += len(line.split())
    print('Total number of tags:', len(tags))
    print('Average number of tags per node:', tagcounts/linecount)
    
            
    
# statistics(URL.FILES.serviceData)

def top20_categories(data):
    labelFile = os.path.join(data, "labels.txt")
    categorynameFile = os.path.join(data, "categoryNames.txt")
    
    cname = dict()
    with open(categorynameFile) as cr:
        for line in cr:
            cname[line.split()[0]] = line.split()[1]
    
    cate = dict()
    with open(labelFile) as lr:
        for line in lr:
            if not line.split()[1] in cate.keys():
                cate[line.split()[1]] = 1
            else:
                cate[line.split()[1]] = cate[line.split()[1]] + 1
    print(len(cate))
    
    categories = [(key, value) for key, value in sorted(cate.items(), key=lambda x: x[1], reverse=True)]
    for k, v in categories[0:40]:
        print(k, ' ', cname[k],' ', v)
    
    
# top20_categories(URL.FILES.serviceData)

def gettopNsimilar(k, mk, mapis, adjlist):
    result = []
    existingapis = []
    if mk in adjlist.keys():
        existingapis = adjlist[mk]
    for api in mapis[mk]:
        if not api in existingapis:
            result.append(api)
        if len(result) >= k:
            break
    return result

# predict the missing and potential composition links between API and Mashup services
# k: number of APIs recommended for each Mashup
def link_prediction(data, k):
    
    adjlistFile = os.path.join(data, "adjlist.txt")
    apimashup_idsFile = os.path.join(data, "apimashup_ids.txt")
    thetaFile = os.path.join(data, "Att-RTM", "models", "service_attrtm_theta.txt")
    mashupapisimilarityFile = os.path.join(data, "service_similarity.txt")
    
    expandedadjlistFile = os.path.join(data, "adjlist_expand"+str(k)+".txt")
    
    adjlist = dict()
    idflag = dict()
    with open(adjlistFile) as ar, open(apimashup_idsFile) as air:
        for line in ar:
            params = line.split()
            adjlist[params[0]] = params[1:]
        for line in air:
            idflag[line.split()[0]] = line.split()[1]
    apis = dict()
    mashups = dict()
    with open(thetaFile) as tr:
        cont = 0
        for line in tr:
            if idflag[str(cont)] == 'A':
                apis[str(cont)] = [float(v) for v in line.split()]
            elif idflag[str(cont)] == 'M': 
                mashups[str(cont)] = [float(v) for v in line.split()]
            cont += 1
    print(len(apis))
    print(len(mashups))
    
    mapis = dict()
    with open(mashupapisimilarityFile) as mr:
        for line in mr:
            params = line.split()
            mapis[params[0]] = params[1:]
    
    for mk in mashups.keys():
        result_apis = gettopNsimilar(k, mk, mapis, adjlist)
        adjlist[mk].extend(result_apis)
        
        for api in result_apis:
            if api in adjlist.keys():
                adjlist[api].append(mk)
            else:
                adjlist[api] = [mk]
    
    with open(expandedadjlistFile, 'w') as ew:
        for k in adjlist.keys():
            line = k +' ' + ' '.join(adjlist[k])
            ew.write(line+'\n')

# link_prediction(URL.FILES.serviceData, 6)  

def topNsimilar(data):   
    from scipy import spatial
    
    adjlistFile = os.path.join(data, "adjlist.txt")
    apimashup_idsFile = os.path.join(data, "apimashup_ids.txt")
    thetaFile = os.path.join(data, "Att-RTM", "models", "service_attrtm_theta.txt")
    mashupapisimilarityFile = os.path.join(data, "service_similarity.txt")
    
    adjlist = dict()
    idflag = dict()
    with open(adjlistFile) as ar, open(apimashup_idsFile) as air:
        for line in ar:
            params = line.split()
            adjlist[params[0]] = params[1:]
        for line in air:
            idflag[line.split()[0]] = line.split()[1]
    apis = dict()
    mashups = dict()
    with open(thetaFile) as tr:
        cont = 0
        for line in tr:
            if idflag[str(cont)] == 'A':
                apis[str(cont)] = [float(v) for v in line.split()]
            elif idflag[str(cont)] == 'M': 
                mashups[str(cont)] = [float(v) for v in line.split()]
            cont += 1
    n = 0
    with open(mashupapisimilarityFile, 'w') as mw:    
        for mk in mashups.keys():
            print(n)    
            mv = mashups[mk]
            sims = dict()
#             ccc = 0
            for ak in apis.keys():
#                 print(ak,' ', ccc)
                av = apis[ak]
                result = 1 - spatial.distance.cosine(mv, av)
                sims[ak] = result
#                 ccc += 1
            api_sims = [key for key, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)]
            line = mk + ' ' + ' '.join(api_sims)
            mw.write(line+'\n')
            n += 1
# topNsimilar(URL.FILES.serviceData)

def cal_performance(actual, predict):
    precision = .0
    recall = .0
    fscore = .0
    
    hitcount = 0
    for p in predict:
        if p in actual:
            hitcount += 1
    recall = hitcount / len(actual)
    precision = hitcount / len(predict)
    
    if recall != 0 and precision != 0:
        fscore = 2 * recall * precision / (recall + precision)
    
    return precision, recall, fscore

def service_recommendation(data, K):
    
    adjlistFile = os.path.join(data, "adjlist.txt")
    apimashup_idsFile = os.path.join(data, "apimashup_ids.txt")
    mashupapisimilarityFile = os.path.join(data, "service_similarity.txt")
    
    adjlist = dict()
    idflag = dict()
    apis = []
    mashups = []
    with open(adjlistFile) as ar, open(apimashup_idsFile) as air:
        for line in air:
            idflag[line.split()[0]] = line.split()[1]
            if line.split()[1] == 'A':
                apis.append(line.split()[0])
            else:
                mashups.append(line.split()[0])
        for line in ar:
            params = line.split()
            adjlist[params[0]] = params[1:]
    
    print(len(apis))
    print(len(mashups))
    
    mapis = dict()
    with open(mashupapisimilarityFile) as mr:
        for line in mr:
            params = line.split()
            mapis[params[0]] = params[1:]
    
    recalls = 0
    precisions = 0
    fscores = 0
    for mashup in mashups:
        actual = adjlist[mashup]    
        predict =  mapis[mashup][0:K]
        precision, recall, fscore = cal_performance(actual, predict)
        
        recalls += recall
        precisions += precision
        fscores += fscore
    recalls = recalls / len(mashups)
    precisions = precisions / len(mashups)
    fscores = fscores / len(mashups)
    
    print('Recall:{}, Precision:{}, F-measure:{}'.format(recalls, precisions, fscores))
        
    
# service_recommendation(URL.FILES.serviceData, 10)    
