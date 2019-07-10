#!/usr/bin/env python3

from dataset import *
from lib import pca, euclidean_distance, cosine_distance
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle

'''
In this classifier, we are storing image feature vectors and its labels.
In the classification stage of testing data, an unlabeled image vector
is classified by assigning the label which is most frequent among k training
image vectors. Here, k is user-defined parameter, we are using K equal to 29.
For classification of testing data, we are using cosine distance to find K-nearest image vectors.

The cosine distance between two vectors a and b is defined as 1 - cosine-similarity(a, b)
and cosine similarity is defined as
cosine-similarity(a, b) = ||a.b||/(||a||.||b||)
where ||x|| is the L2 norm of the vector x.

While Euclidean distance measures the magnitude of distance between the two vectors,
Cosine distance measures the angle between the two vectors.
Thus, cosine distance would be more robust to brightness and contrast changes
than Euclidean distance.
'''

class KNearestNeighborsClassifier:
    def __init__(self, k=3, n_features=10, distance=None):
        self.k = k
        self.n_features = n_features
        self.distance = distance if distance is not None else euclidean_distance
        self.X = None
        self.X_mean = None
        self.y = None
        self.V = None

    def fit(self, X, y):
        self.X_mean, self.V = pca(X, n_components=self.n_features)
        self.X = np.dot(X, self.V)
        self.y = y

    def predict(self, x):
        x = x[np.newaxis, :]
        x = x - self.X_mean
        x = np.dot(x, self.V)

        distances = self.distance(x, self.X).ravel()

        top_k = (distances).argsort()[:self.k]
        top_k_classes = [self.y[each] for each in top_k]
        [(prediction, count)] = Counter(top_k_classes).most_common(1)
        return prediction




def select_k_param(train, test):
    #Plot graph for the following values of k
    k_values = list(np.linspace(10, 192, num = 100, endpoint = True, dtype = 'int'))
    error_dict = {}
    for k_val in k_values:
        error_dict[k_val] = list()
        model = KNearestNeighborsClassifier(k=k_val, n_features=64, distance=cosine_distance)
        model.fit(train.pixel_values, train.orientations)
        correct = 0
        for X, y in zip(test.pixel_values, test.orientations):
            if model.predict(X) == y:
                correct += 1
        error_dict[k_val].append(correct/test.size)

    plt_accuracy = []
    plt_x = []
    plt_std_deviation = []

    #Calculating mean and standard deviation of the outcome
    for key in error_dict:
        error_dict[key] = [np.mean(error_dict[key]),np.std(error_dict[key])]
        plt_accuracy.append(error_dict[key][0])
        plt_x.append(key)
        plt_std_deviation.append(error_dict[key][1])

    plt.errorbar(plt_x, plt_accuracy, yerr = plt_std_deviation, fmt ='--o', color = 'red', label="knn" )
    plt.title("K vs accuracy")
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.legend()
    figure_name = "knn_k_param.png"
    plt.savefig(figure_name)


def gen_learning_curve(train_data, test):
    learning_curve_size = list(np.linspace(0.1, 1, num = 10, endpoint = True, dtype = 'float'))
    total_iter = 20
    error_dict = {}
    train, _ = train_test_split(train_data, fraction = data_size)

    for data_size in learning_curve_size:
        error_dict[data_size] = list()
        #execute code 20 times for each training size to rule out random sampling errors
        for i in range(0, total_iter):
            print("Size of data: ", train.size)
            model = KNearestNeighborsClassifier(k=int(np.sqrt(train.size)), n_features=64, distance=cosine_distance)
            model.fit(train.pixel_values, train.orientations)
            correct = 0

            for X, y in zip(test.pixel_values, test.orientations):
                if model.predict(X) == y:
                    correct += 1
            error_dict[data_size].append(correct/test.size)
    plt_accuracy = []
    plt_x = []
    plt_std_deviation = []

    #Calculating mean and standard deviation of the outcome
    for key in error_dict:
        error_dict[key] = [np.mean(error_dict[key]),np.std(error_dict[key])]

        plt_accuracy.append(error_dict[key][0])
        plt_x.append(key)
        plt_std_deviation.append(error_dict[key][1])
    #plot the error bars for the learning curve and save the image
    plt.errorbar(plt_x, plt_accuracy, yerr = plt_std_deviation, fmt ='--o', color = 'red', label="knn" )
    plt.title("Learning curves")
    plt.xlabel("Proportion of training points")
    plt.ylabel("Accuracy")
    plt.legend()
    figure_name = "knn_learning_curve.png"
    plt.savefig(figure_name)


def knn_predict(test_or_train, test_file, model_file):
    if(test_or_train == 'test'):
        test = read_data(test_file)
        with open(model_file, "rb") as handle:
            model = pickle.load(handle)
        predicted_values = []
        correct = 0
        for X, y in zip(test.pixel_values, test.orientations):
            predicted_values.append(model.predict(X))
            if predicted_values[-1] == y:
                correct += 1
        return list(zip(test.image_names, predicted_values))
    else:
        train = read_data("train-data.txt")
        test = read_data("test-data.txt")
        model = KNearestNeighborsClassifier(k=29, n_features=75, distance=cosine_distance)
        model.fit(train.pixel_values, train.orientations)

        with open(model_file, "wb") as handle:
            pickle.dump(model, handle)
        return []

if __name__ == "__main__":
    train = read_data("train-data.txt")
    test = read_data("test-data.txt")

    model = KNearestNeighborsClassifier(k=29, n_features=75, distance=cosine_distance)
    model.fit(train.pixel_values, train.orientations)

    with open("knn_model.txt", "wb") as handle:
        pickle.dump(model, handle)

    with open("knn_model.txt", "rb") as handle:
        model = pickle.load(handle)

    correct = 0
    #gen_learning_curve(train, test)
    #select_k_param(train, test)

    for X, y in zip(test.pixel_values, test.orientations):
        if model.predict(X) == y:
            correct += 1
            print (correct)
    print (correct / test.size)
