import numpy as np
import os
import gensim
import URLs as URL
import random


def load_data(dir):
    
    doc_ids = []
    doc_cnt = []
    voca = []
    
    content_token = os.path.join(dir, "rtm_content_token.txt")
    adjlist = os.path.join(dir, "rtm_adjlist.txt")
    
    with open(content_token, 'r', encoding="iso-8859-1") as cr:
        
        for cline in cr:
            ids = []
            cnt = []
#             words = gensim.parsing.stem_text(cline).split()
            words = cline.split()
            for word in words:
                if not word in voca:
                    voca.append(word)
                word_index = voca.index(word)
                if not word_index in ids:
                    ids.append(word_index)
                    cnt.append(1)
                else:
                    word_id = ids.index(word_index)
                    cnt[word_id] = cnt[word_id] + 1
            if len(words) == 0:
                ids.append(random.randint(1,len(voca)))
                cnt.append(1)
            
            doc_ids.append(ids)
            doc_cnt.append(cnt)
    
    doc_link_ids = dict()
    with open(adjlist, 'r') as ar:
        for line in ar:
            params = line.split()
            doc_id = int(params[0])
            link_ids = [int(value) for value in params[1:]]
            doc_link_ids[doc_id] = link_ids
    
    doc_links = []
    n_doc = len(doc_ids)
    for i in range(n_doc):
        if not i in doc_link_ids.keys():
            doc_links.append([])
        else:
            doc_links.append(doc_link_ids[i])
    
    return doc_ids, doc_cnt, doc_links, voca

# doc_ids, doc_cnt, doc_links, voca = load_data(URL.FILES.coraData)

def sampling_from_dist(prob):
    """ Sample index from a list of unnormalised probability distribution
        same as np.random.multinomial(1, prob/np.sum(prob)).argmax()

    Parameters
    ----------
    prob: ndarray
        array of unnormalised probability distribution

    Returns
    -------
    new_topic: return a sampled index
    """
    thr = prob.sum() * np.random.rand()
    new_topic = 0
    tmp = prob[new_topic]
    while tmp < thr:
        new_topic += 1
        tmp += prob[new_topic]
    return new_topic


def sampling_from_dict(prob):
    """ sample key from dictionary `prob` where values are unnormalised probability distribution

    Parameters
    ----------
    prob: dict
        key = topic
        value = unnormalised probability of the topic

    Returns
    -------
    key: int
        sampled key
    """
    prob_sum = sum(prob.values())

    thr = prob_sum * np.random.rand()
    tmp = 0
    for key, p in prob.items():
        tmp += p
        if tmp < thr:
            new_topic = key
    return new_topic


def isfloat(value):
    """
    Check the value is convertable to float value
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_voca(path):
    """
    open file from path and read each line to return the word list
    """
    with open(path, 'r') as f:
        return [word.strip() for word in f.readlines()]


def word_cnt_to_bow_list(word_ids, word_cnt):
    corpus_list = list()
    for di in range(len(word_ids)):
        doc_list = list()
        for wi in range(len(word_ids[di])):
            word = word_ids[di][wi]
            for c in range(word_cnt[di][wi]):
                doc_list.append(word)
        corpus_list.append(doc_list)
    return corpus_list


def log_normalize(log_prob_vector):
    """
    returns a probability vector of log probability vector
    """
    max_v = log_prob_vector.max()
    log_prob_vector += max_v
    log_prob_vector = np.exp(log_prob_vector)
    log_prob_vector /= log_prob_vector.sum()
    return log_prob_vector


def convert_cnt_to_list(word_ids, word_cnt):
    corpus = list()

    for di in range(len(word_ids)):
        doc = list()
        doc_ids = word_ids[di]
        doc_cnt = word_cnt[di]
        for wi in range(len(doc_ids)):
            word_id = doc_ids[wi]
            for si in range(doc_cnt[wi]):
                doc.append(word_id)
        corpus.append(doc)
    return corpus


def write_top_words(topic_word_matrix, vocab, filepath, n_words=20, delimiter=',', newline='\n'):
    with open(filepath, 'w') as f:
        for ti in range(topic_word_matrix.shape[0]):
            top_words = vocab[topic_word_matrix[ti, :].argsort()[::-1][:n_words]]
            f.write('%d' % (ti))
            for word in top_words:
                f.write(delimiter + word)
            f.write(newline)


def get_top_words(topic_word_matrix, vocab, topic, n_words=20):
    if not isinstance(vocab, np.ndarray):
        vocab = np.array(vocab)
    top_words = vocab[topic_word_matrix[topic].argsort()[::-1][:n_words]]
    return top_words


def convertDataFormat_tortm(dir):
    
    cntFile = os.path.join(dir,"content_token.txt")
    graphFile = os.path.join(dir, "adjlist.txt")
    labelFile = os.path.join(dir, "labels.txt")
    
    content_token = os.path.join(dir, "rtm_content_token.txt")
    labels = os.path.join(dir, "rtm_labels.txt")
    adjlist = os.path.join(dir, "rtm_adjlist.txt")
    
    docid_dict = dict()
    with open(cntFile, 'r') as dr, open(content_token, 'w') as cw:
        cont = 0
        for line in dr:
            words = line.split()
            if len(words) == 0:
                cont += 1
                continue
            docid_dict[str(cont)] = len(docid_dict)
            cont += 1
            
            cw.write(' '.join(words)+"\n")
            
    label_dict = dict()
    with open(labelFile, 'r') as lr, open(labels, 'w') as lw:
         
        for line in lr:
            params = line.split()
            if not params[0] in docid_dict.keys():
                continue
            nodeid = docid_dict[params[0]]
            if not params[1] in label_dict.keys():
                label_dict[params[1]] = 'L' + str(len(label_dict))
            lw.write(str(nodeid) + ' ' + label_dict[params[1]] + '\n')
            
    adjnodes = dict()
    with open(graphFile, 'r') as gr, open(adjlist, 'w') as aw:
        for line in gr:
            ns = line.split()
            
            nodes = []
            for v in ns:
                if not v in docid_dict.keys():
                    continue
                nodes.append(str(docid_dict[v]))
        
            aw.write(' '.join(nodes) + '\n')

# convertDataFormat_tortm(URL.FILES.serviceData)