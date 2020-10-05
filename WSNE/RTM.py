from __future__ import print_function
import time

import numpy as np
from scipy.special import gammaln, psi
import rtmutil
from rtmutil import write_top_words
from formatted_logger import formatted_logger

import logging
import os

logger = logging.getLogger('RelationalTopicModel')
logger.propagate=False
import URLs as URL


eps = 1e-20

logger = formatted_logger('RelationalTopicModel', 'info')


class RelationalTopicModel:
    """ implementation of relational topic models by Chang and Blei (2009)
    use the exponential link probability function in here

    Attributes
    ----------
    eta: ndarray, shape (n_topic)
        coefficient of exponential function
    rho: int
        pseudo number of negative example
    """

    def __init__(self, n_topic, n_doc, n_voca, alpha=0.1, rho=1000, **kwargs):
        self.n_doc = n_doc
        self.n_topic = n_topic
        self.n_voca = n_voca

        self.alpha = alpha

        self.gamma = np.random.gamma(100., 1. / 100, [self.n_doc, self.n_topic])
        self.beta = np.random.dirichlet([5] * self.n_voca, self.n_topic)

        self.nu = 0
        self.eta = np.random.normal(0., 1, self.n_topic)

        self.phi = list()
        self.pi = np.zeros([self.n_doc, self.n_topic])

        self.rho = rho

        self.verbose = kwargs.pop('verbose', True)

        logger.info('Initialize RTM: num_voca:%d, num_topic:%d, num_doc:%d' % (self.n_voca, self.n_topic, self.n_doc))

    def fit(self, doc_ids, doc_cnt, doc_links, max_iter=100):
        for di in range(self.n_doc):
            unique_word = len(doc_ids[di])
            cnt = doc_cnt[di]
            self.phi.append(np.random.dirichlet([10] * self.n_topic, unique_word).T)  # list of KxW
            self.pi[di, :] = np.sum(cnt * self.phi[di], 1) / np.sum(cnt * self.phi[di])

        for iter in range(max_iter):
            tic = time.time()
            self.variation_update(doc_ids, doc_cnt, doc_links)
            self.parameter_estimation(doc_links)
            if self.verbose:
                elbo = self.compute_elbo(doc_ids, doc_cnt, doc_links)
                logger.info('[ITER] %3d,\tElapsed time: %.3f\tELBO: %.3f', iter, time.time()-tic, elbo)

    def compute_elbo(self, doc_ids, doc_cnt, doc_links):
        """ compute evidence lower bound for trained models
        """
        elbo = 0

        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]  # D x K
        log_beta = np.log(self.beta + eps)

        for di in range(self.n_doc):
            words = doc_ids[di]
            cnt = doc_cnt[di]

            elbo += np.sum(cnt * (self.phi[di] * log_beta[:, words]))  # E_q[log p(w_{d,n}|\beta,z_{d,n})]
            elbo += np.sum((self.alpha - 1.) * e_log_theta[di, :])  # E_q[log p(\theta_d | alpha)]
            elbo += np.sum(self.phi[di].T * e_log_theta[di, :])  # E_q[log p(z_{d,n}|\theta_d)]

            elbo += -gammaln(np.sum(self.gamma[di, :])) + np.sum(gammaln(self.gamma[di, :])) \
                    - np.sum((self.gamma[di, :] - 1.) * (e_log_theta[di, :]))  # - E_q[log q(theta|gamma)]
            elbo += - np.sum(cnt * self.phi[di] * np.log(self.phi[di]))  # - E_q[log q(z|phi)]

            for adi in doc_links[di]:
                elbo += np.dot(self.eta,
                               self.pi[di] * self.pi[adi]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]

        return elbo

    def variation_update(self, doc_ids, doc_cnt, doc_links):
        # update phi, gamma
        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]
#         print("self.gamma[0:10]:", self.gamma[0:10])
#         print("psi(self.gamma)[0:10]:", psi(self.gamma)[0:10])
#         print("np.sum(self.gamma, 1)[0:10]:", np.sum(self.gamma, 1)[0:10])
#         print("psi(np.sum(self.gamma, 1))[0:10]:", psi(np.sum(self.gamma, 1))[0:10])
#         print("psi(np.sum(self.gamma, 1)[0:10])[:, np.newaxis]:", psi(np.sum(self.gamma, 1)[0:10])[:, np.newaxis])
#         print("e_log_theta[0:10]:", e_log_theta[0:10])
#         print("e_log_theta shape:", e_log_theta.shape)
        new_beta = np.zeros([self.n_topic, self.n_voca])
        
#         print("np.log(self.beta[:, doc_ids[0]] + eps):", np.log(self.beta[:, doc_ids[0]] + eps))
#         print("e_log_theta[0, :][:, np.newaxis]:", e_log_theta[0, :][:, np.newaxis])
        

        for di in range(self.n_doc):
            words = doc_ids[di]
            cnt = doc_cnt[di]
            doc_len = np.sum(cnt)
            
            new_phi = np.log(self.beta[:, words] + eps) + e_log_theta[di, :][:, np.newaxis]
            
            gradient = np.zeros(self.n_topic)
            for ai in doc_links[di]:
                gradient += self.eta * self.pi[ai, :] / doc_len

            new_phi += gradient[:, np.newaxis]
            new_phi = np.exp(new_phi)
            new_phi = new_phi / np.sum(new_phi, 0)

            self.phi[di] = new_phi

            self.pi[di, :] = np.sum(cnt * self.phi[di], 1) / np.sum(cnt * self.phi[di])
            self.gamma[di, :] = np.sum(cnt * self.phi[di], 1) + self.alpha
            new_beta[:, words] += (cnt * self.phi[di])

        self.beta = new_beta / np.sum(new_beta, 1)[:, np.newaxis]

    def parameter_estimation(self, doc_links):
        # update eta, nu
        pi_sum = np.zeros(self.n_topic)

        num_links = 0.

        for di in range(self.n_doc):
            for adi in doc_links[di]:
                pi_sum += self.pi[di, :] * self.pi[adi, :]
                num_links += 1

        num_links /= 2.  # divide by 2 for bidirectional edge
        pi_sum /= 2.

        pi_alpha = np.zeros(self.n_topic) + self.alpha / (self.alpha * self.n_topic) * self.alpha / (self.alpha * self.n_topic)

        self.nu = np.log(num_links - np.sum(pi_sum)) - np.log(
            self.rho * (self.n_topic - 1) / self.n_topic + num_links - np.sum(pi_sum))
        self.eta = np.log(pi_sum) - np.log(pi_sum + self.rho * pi_alpha) - self.nu

    def save_model(self, output_directory, experiment_count, vocab=None):
        np.savetxt(os.path.join(output_directory, 'RTM', str(experiment_count)+'eta.txt'), self.eta, delimiter='\t')
        np.savetxt(os.path.join(output_directory, 'RTM',  str(experiment_count) + 'beta.txt'), self.beta, delimiter='\t')
#         np.savetxt(output_directory + '/gamma.txt', self.gamma, delimiter='\t')
        np.savetxt(os.path.join(output_directory, 'RTM',  str(experiment_count) + 'node_embeddings.txt'), self.gamma, delimiter='\t')
        with open(os.path.join(output_directory, 'RTM',  str(experiment_count) + 'nu.txt'), 'w') as f:
            f.write('%f\n' % self.nu)

        if vocab is not None:
            write_top_words(self.beta, vocab, os.path.join(output_directory, 'RTM', str(experiment_count)+'top_words.csv'))

def evaluate(dataset, experiment_count, train_size, experiment_num):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    from sklearn.cross_validation import train_test_split
    # prepare the data for node classification
    selectedCarfile = os.path.join(dataset, "top20category.txt")
    top20categories = []
    with open(selectedCarfile) as sr:
        for line in sr:
            top20categories.append(line.split()[0])
    
    selectedembedfile = os.path.join(dataset, "rtm_adjlist.txt")
    uniquenodes = []
    with open(selectedembedfile) as sr:
        for line in sr:
            for n in line.split():
                if not n in uniquenodes:
                    uniquenodes.append(n)
    print(len(uniquenodes))
    # prepare the data for node classification
    all_data = dict()
    labelsfile = os.path.join(dataset, "rtm_labels.txt")
    with open(labelsfile) as lr:
        for line in lr:
            params = line.split()
            if params[1] in top20categories and params[0] in uniquenodes: 
                all_data[params[0]] = params[1]
        
#     train, test = train_test_split(list(all_data.keys()), train_size=train_size, random_state=1)
    
#     print(len(all_data))
#     print(len(train))
    
    allvecs = dict()
    node_embed_file = os.path.join(dataset, 'RTM', str(experiment_count) + 'node_embeddings.txt')
    with open(node_embed_file) as er:
        line_count = 0
        for line in er:
            params = line.split()
            if not str(line_count) in all_data.keys():
                line_count+=1
                continue            
            node_embeds = [float(value) for value in params]
            allvecs[str(line_count)] = node_embeds
            line_count+=1

    
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
    
    dataset = URL.FILES.serviceData
#     doc_ids, doc_cnt, doc_links, voca = rtmutil.load_data(dataset)
     
#     print(doc_ids[0:2])
#     print(doc_cnt[0:2])
#     print(doc_links[0:2])
#     print(voca[0:2])
#           
#     n_doc = len(doc_ids)
#     n_topic = 100
#     n_voca = len(voca)
#     max_iter = 50
#         
    train_size = 0.7
    experiment_num = 20 # number of evaluation times for each train size, the average result and standard deviation are calculated
    experiment_count=1
#       
#     models = RelationalTopicModel(n_topic, n_doc, n_voca, verbose=True)
#     models.fit(doc_ids, doc_cnt, doc_links, max_iter=max_iter)
#     models.save_model(dataset, experiment_count)
    
    evaluate(dataset, experiment_count, train_size, experiment_num)
    
if __name__ == '__main__':
    main() 
