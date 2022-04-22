from mynlplib.constants import OFFSET
import numpy as np
import torch

# deliverable 6.1
def get_top_features_for_label(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''
    dic = {k:v for k,v in weights.items() if k[0] == label}
    lis2 = [(k,v) for k,v in sorted(dic.items(), key = lambda item : item[1], reverse=True)]
    
    return lis2[:k]                             
    #raise NotImplementedError

# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''
    
    vocab = sorted(vocab)
    for name,parameter in model.named_parameters():
        if name == "Linear.weight":
            weights = parameter
    for i in range(len(label_set)):
        if label_set[i] == label:
            idx = i
    lis = []
    idxs = np.argsort(weights[idx].detach().numpy())[-k:]
    for i in range(len(idxs)-1, -1, -1):
        lis.append(vocab[idxs[i]])
    
    return lis

    raise NotImplementedError

# deliverable 7.1
def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    lis = [counts[i] for i in range(len(counts)) if counts[i] != 0]
    if len(lis) == 0:
        return 0
    else:
        return sum(lis)/len(lis)
    
    raise NotImplementedError

# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    new_data = np.zeros((data.shape[0], data.shape[1]+7))
    for i in range(len(data)):
        lis = np.zeros(7)
        ratio = get_token_type_ratio(data[i])
        if ratio >= 6:
            lis[-1] = 1
        else:
            lis[int(np.floor(ratio))] = 1
        new_data[i] = np.concatenate((data[i], lis))
        
    return new_data     
            
    
    raise NotImplementedError
