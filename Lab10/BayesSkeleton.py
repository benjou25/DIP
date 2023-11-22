import numpy as np

import pickle


class Data:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label


def prepare_data(data):
    labels = np.unique(np.array([dat.label for dat in data]))
    class_dim = len(labels)
    features = [[] for i in range(class_dim)]
    for dat in data:
        features[dat.label].append(dat.feature)
    return labels, features


def train(train_data):
    # Return
    # mean ... list with one entry per class
    #          each entry is the mean of the feature vectors of a class
    # covariance ... list with one entry per class
    #          each entry is the covariance of the feature vectors of a class
    labels, features = prepare_data(train_data)
    mean = []
    covariance = []
    for label in labels:
        feature_matrix = np.array(features[label])
        mean.append(np.mean(feature_matrix, axis=0))
        covariance.append(np.cov(feature_matrix, rowvar=False))
    return mean, covariance


def evaluateCost(feature_vector, m, c):
    # Input
    # feature_vector ... feature vector under test
    # m     mean of the feature vectors for a class
    # c     covariance of the feature vectors of a class
    # Output
    #   some scalar proportional to the logarithm of the probability d_j(feature_vector)
    # Mahalanobis distance calculation
    diff = feature_vector - m
    inv_covariance = np.linalg.inv(c)
    mahalanobis_dist = np.dot(np.dot(diff, inv_covariance), diff.T)
    return mahalanobis_dist


def classify(test_data, mean, covariance):
    labels, features = prepare_data(test_data)
    decisions = []

    for label in labels:
        class_costs = []
        feature_matrix = np.array(features[label])
        for j in range(len(mean)):
            # Evaluate cost for each feature vector in the class
            costs = [evaluateCost(feature_vector, mean[j], covariance[j]) for feature_vector in feature_matrix]
            print('\n',costs)
            # Use the minimum cost among all feature vectors in the class
            class_costs.append(min(costs))

        # Make a decision based on the minimum cost
        decision = np.argmin(class_costs)
        decisions.append(decision)

    return decisions





def computeConfusionMatrix(decisions, test_data):
    pass
    pass
    pass


def main():
    train_data = pickle.load(open("train_data.pkl", "rb"))
    test_data = pickle.load(open("test_data.pkl", "rb"))

    # Train: Compute mean and covariance for each object class from {0,1,2,3}
    # returns one list entry per object class
    mean, covariance = train(train_data)

    for i in range(4):
        print(f"\nmean {i+1}:", mean[i])
        print(f"covar {i+1}:", covariance[i])
    
    # Decide: Compute decision for each feature vector from test_data
    # return a list of class indices from the set {0,1,2,3}
    decisions = classify(test_data, mean, covariance)
    print(decisions)
    
    # Copmute the confusion matrix
    confusion_matrix = computeConfusionMatrix(decisions, test_data)
    print(confusion_matrix)

if __name__ == "__main__":
    main()
