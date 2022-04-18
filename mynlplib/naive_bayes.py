from mynlplib.constants import OFFSET
from mynlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict
from collections import Counter

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    def ret():
        return 0
    d = defaultdict(ret)
    for i in range(len(y)):
        if y[i] == label:
            for word in x[i]:
                d[word] = d.get(word,0) + x[i][word]
    return d                
    #raise NotImplementedError

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    def ret():
        return 0
    ans = defaultdict(ret)
    d = get_corpus_counts(x,y,label)
    for word in vocab:
        ans[word] = np.log((d[word] + smoothing)/(sum(d.values()) + smoothing*len(vocab))) 
        
    return ans    
    #raise NotImplementedError

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    def ret():
        return 0
    w = defaultdict(ret)
    
    l_counts = Counter(y)
    vocab = set()
    
    for counts in x:
        for word in counts:
            vocab.add(word)
            
    for label in l_counts:
        c = l_counts[label]/sum(l_counts.values())
        #print(label,c)
        d = estimate_pxy(x,y,label,smoothing,vocab)
        for word,prob in d.items():
            w[(label,word)] = d[word]
        
        w[(label, OFFSET)] = np.log(c)
           
    return w
    #raise NotImplementedError

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    '''
    labels = set(y_tr)
    dic = {}
    for s in smoothers:
        weights = estimate_nb(x_tr, y_tr,s)
        preds = clf_base.predict_all(x_dv, weights, labels)
        score = evaluation.acc(preds,y_dv)
        dic[s] = score
    
    return (max(dic, key = dic.get)), dic

    #raise NotImplementedError
