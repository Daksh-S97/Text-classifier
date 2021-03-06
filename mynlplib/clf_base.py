from mynlplib.constants import OFFSET
import numpy as np

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# This will no longer work for our purposes since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

# deliverable 2.1
def make_feature_vector(base_features,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    dic = {}
    for word,count in base_features.items():
        dic[(label, word)] = count
    
    dic[(label, OFFSET)] = 1
    return dic

    #raise NotImplementedError

# deliverable 2.2
def predict(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''
    dic = dict()
        
    if len(weights)==0:
        for label in labels:
            dic[label] = 0.0
            
    for features, weight in weights.items():
        if features[1] == '**OFFSET**':
            dic[features[0]] = dic.get(features[0], 0) + weight
        dic[features[0]] = dic.get(features[0], 0) + weight*base_features.get(features[1],0) 
    
    lab = argmax(dic)
    return lab,dic
    #raise NotImplementedError

def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat