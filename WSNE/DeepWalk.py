import tensorflow as tf
import math
import numpy as np
import URLs as URL
import os
import deepwalkutil
import time

class DeepWalk:
            
    def __init__(self, dataset, experiment_count, num_skip, node_window_size, learning_rate,node_node_batch_size,node_size, node_embsize, loss_type = 'nce_loss',
                  optimize = 'Adagrad', num_sampled=5,num_runs=100000):

        graph = tf.Graph()
            
        with graph.as_default():
            # Input data
            train_inputs = tf.placeholder(tf.int32, shape=[node_node_batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[node_node_batch_size, 1])
            embeddings = tf.Variable(
                    tf.random_uniform([node_size, node_embsize], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
            # Construct the variables
            nce_weights = tf.Variable(
                    tf.truncated_normal([node_size, node_embsize],
                        stddev=1.0 / math.sqrt(node_embsize)))
            nce_biases = tf.Variable(tf.zeros([node_size]))
            # compute the loss with negative sampling
            if loss_type == 'sampled_softmax_loss':
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs = embed,
                    num_sampled=num_sampled,
                    num_classes=node_size))
            elif loss_type == 'nce_loss':
                loss = tf.reduce_mean(tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs = embed,
                    num_sampled=num_sampled,
                    num_classes=node_size))
        
            # Optimizer.
            if optimize == 'Adagrad':
                
                global_step = tf.Variable(1, name="global_step", trainable=False)
                optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
#                 optimizer1 = tf.train.AdamOptimizer(learning_rate)
#                 grads_and_vars = optimizer1.compute_gradients(loss)
#                 grads, _ = list(zip(*grads_and_vars))
#                 tf.summary.scalar("gradient_norm", tf.global_norm(grads))
#                 optimizer = optimizer1.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step,
#                                                           name="train_op")
                
            elif optimize == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
            norm_node_embeddings = embeddings / tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        
            # Add variable initializer
            init = tf.global_variables_initializer()  
        
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            with tf.Session(graph=graph,config=config) as sess:
                init.run()
                print("Initialized!")
                average_loss = 0
        
                nn_training_batches, reverse_node_dict= deepwalkutil.generate_batch_deepwalk(dataset, node_node_batch_size, 
                                                           num_skip, node_window_size)
                num_runs = num_runs
                num_out = 10000 # how many to output the loss
                start = time.time()
                batch_count = 0
        
                for i in range(num_runs):
                    nn_batch = nn_training_batches[i%len(nn_training_batches)]
        
                    nn_batch_inputs = np.array(nn_batch[0])
                    nn_batch_lables = np.expand_dims(np.array(nn_batch[1]), axis =1)
                    
#                     shuffle_indices = np.random.permutation(np.arange(len(nn_batch_inputs)))
#                     nn_batch_inputs = nn_batch_inputs[shuffle_indices]
#                     nn_batch_lables = nn_batch_lables[shuffle_indices]
                    
                    feed_dicts = {train_inputs: nn_batch_inputs, train_labels: nn_batch_lables}
                    # run the graph
                    sess.run(optimizer, feed_dict=feed_dicts)
                    loss_val = sess.run(loss, feed_dict=feed_dicts)
                    average_loss += loss_val
                    batch_count += 1
        
                    if i % num_out == 0:
                        average_loss = average_loss / batch_count
                        print("num runs = {}, average loss = {}".format(i, average_loss))
        
                norm_node_embeddings = sess.run(norm_node_embeddings)
        
                end = time.time()
                print('time used:', (end-start))



            # Save embeddings to local disk
            node_embed_file = os.path.join(dataset, 'deepwalk', str(experiment_count)+'node_embeddings.txt')
            
            with open(node_embed_file, 'w') as nw:
                line = str(norm_node_embeddings.shape[0]) + " " + str(norm_node_embeddings.shape[1])
                nw.write(line+'\n')
                node_dict_id = 0
                for node_embed in norm_node_embeddings:
                    id1 = reverse_node_dict[node_dict_id]
                    line = str(id1)
                    for e in node_embed:
                        line += ' ' + str(e)
                    nw.write(line + '\n')
                    node_dict_id += 1
            print('Node embedding saved, %d nodes in total.' % node_dict_id)

def evaluate(dataset, experiment_count, train_size, experiment_num):

    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    from sklearn.cross_validation import train_test_split


    selectedCarfile = os.path.join(dataset, "top20category.txt")
    top20categories = []
    with open(selectedCarfile) as sr:
        for line in sr:
            top20categories.append(line.split()[0])
    
    # prepare the data for node classification
    all_data = dict()
    labelsfile = os.path.join(dataset, "labels.txt")
    with open(labelsfile) as lr:
        for line in lr:
            params = line.split()
            if params[1] in top20categories:
                all_data[params[0]] = params[1]
    
#     train, test = train_test_split(list(all_data.keys()), train_size=train_size, random_state=1)
    
#     print(len(all_data))
#     print(len(train))
    
    allvecs = dict()
    node_embed_file = os.path.join(dataset, 'deepwalk', str(experiment_count)+'node_embeddings.txt')
    with open(node_embed_file) as er:
        er.readline()
        for line in er:
            params = line.split()
            node_id = params[0]
            if not node_id in all_data.keys():
                continue
            node_embeds = [float(value) for value in params[1:]]
            allvecs[node_id] = node_embeds
    
    macro_f1s = []
    micro_f1s = []
    for experiment_time in range(experiment_num):
        train, test = train_test_split(list(allvecs.keys()), train_size=train_size, random_state=experiment_time)
        train_vec = []
        train_y = []
        test_vec = []
        test_y = []
        for train_i in train:
            train_vec.append(allvecs[train_i])
            train_y.append(all_data[train_i])
        for test_i in test:
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
#     dataset = URL.FILES.wikiData
    dataset = URL.FILES.serviceData
    num_skip = 4
    node_window_size = 4
    learning_rate = 0.05
    node_node_batch_size = 120
    node_embsize = 100
    num_sampled = 5
    num_runs = 200000
    loss_type = 'nce_loss'
    optimize = 'Adagrad'
    node_size, _ = deepwalkutil.repository_size(dataset, 20)  
    print('node_size:{}'.format(node_size))
    
    
    experiment_count = 1
    experiment_num = 20 # number of evaluation times for each train size, the average result and standard deviation are calculated    
    train_size = 0.1
#     
#     deepwalk = DeepWalk(dataset, experiment_count, num_skip, node_window_size, learning_rate,node_node_batch_size,node_size, node_embsize, loss_type,
#                         optimize, num_sampled, num_runs)
    
    print('experiment_count:{}, train_size:{}'.format(experiment_count, train_size))
    evaluate(dataset, experiment_count, train_size, experiment_num)
if __name__ == '__main__':
    main() 

