import numpy as np
import URLs as URL
import os

def evaluate(dataset, train_size, experiment_num):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    from sklearn.cross_validation import train_test_split
    # prepare the data for node classification
    all_data = dict()
    labelsfile = os.path.join(dataset, "labels.txt")
    with open(labelsfile) as lr:
        for line in lr:
            params = line.split()
            for lab in params[1].split(','):
               all_data[params[0]] = lab
    
#     print(len(all_data))
#     print(len(train))
    
    allvecs = dict()
    # node_embed_file = os.path.join(dataset, 'node_embeddings.txt')
    node_embed_file = os.path.join(dataset, 'PLANE', 'node_embeddings.txt')
    with open(node_embed_file) as er:
        er.readline()
        for line in er:
            params = line.split()
            node_id = params[0]
            node_embeds = [float(value) for value in params[1:]]
            allvecs[node_id] = node_embeds
    
    macro_f1s = []
    micro_f1s = []
    for experiment_time in range(experiment_num):
        train, test = train_test_split(list(all_data.keys()), train_size=train_size, random_state=experiment_time)
        train_vec = []
        train_y = []
        test_vec = []
        test_y = []
        for train_i in train:
            if not train_i in allvecs.keys():
                continue
            train_vec.append(allvecs[train_i])
            train_y.append(all_data[train_i])
        for test_i in test:
            if not test_i in allvecs.keys():
                continue
            test_vec.append(allvecs[test_i])
            test_y.append(all_data[test_i])
        
        classifier = LinearSVC()
        classifier.fit(train_vec, train_y)
        y_pred = classifier.predict(test_vec)
        
        cm = confusion_matrix(test_y, y_pred)
#         print(cm)
        
        acc = accuracy_score(test_y, y_pred)
        macro_recall = recall_score(test_y, y_pred,  average='macro')
        macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
        micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')
        
        macro_f1s.append(macro_f1)
        micro_f1s.append(micro_f1)
        
        print('experiment_time: %d, Classification Accuracy=%f, macro_recall=%f macro_f1=%f, micro_f1=%f' % (experiment_time, acc, macro_recall, macro_f1, micro_f1))              
        
    import statistics
    average_macro_f1 = statistics.mean(macro_f1s)
    average_micro_f1 = statistics.mean(micro_f1s)
    stdev_macro_f1 = statistics.stdev(macro_f1s)
    stdev_micro_f1 = statistics.stdev(micro_f1s)
    
    print('Total experiment time: %d, average_macro_f1=%f, stdev_macro_f1=%f, average_micro_f1=%f, stdev_micro_f1=%f' % (experiment_num, average_macro_f1, stdev_macro_f1, average_micro_f1, stdev_micro_f1))

def main():
    dataset = URL.FILES.citeseerData
    experiment_num = 20 # number of evaluation times for each train size, the average result and standard deviation are calculated    
    train_size = 0.1
    evaluate(dataset, train_size, experiment_num)
    
if __name__ == '__main__':
    main()
    