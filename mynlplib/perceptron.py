from collections import defaultdict
from mynlplib.clf_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    update = defaultdict(float)
    fv2 = make_feature_vector(x,y)
    if len(weights)==0:
        for feature in fv2:
            weights[feature] = fv2[feature]
        
    label, _ = predict(x,weights,labels)
    if label != y:
        fv = make_feature_vector(x,label)
        for feature in list(fv2.keys())+list(fv.keys()):
            update[feature] = fv2.get(feature,0.0) - fv.get(feature,0.0)    
    #raise NotImplementedError

    return update

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its+1):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            update = perceptron_update(x_i,y_i,weights,labels)
            for feature,weight in update.items():
                weights[feature] += weight
            #raise NotImplementedError
            
        weight_history.append(weights.copy())
    return weights, weight_history
