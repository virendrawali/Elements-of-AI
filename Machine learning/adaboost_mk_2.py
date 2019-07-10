#!/usr/bin/env python3

from collections import namedtuple
import numpy as np
from collections import Counter
from lib import entropy, pca
from tqdm import trange
from tree import information_gain
from dataset import read_data, train_test_split
import pickle

Hypothesis = namedtuple("Hypothesis", ["attribute", "value", "labels"])
LEFT = 0
RIGHT = 1

'''
Adaptive boosting is a data classification algorithm whose main focus is to classify
data by converting a number of weak classifier into a strong one.
Decision stumps are used to classify data. In this multiclass adaboost implementation,
we are using one-vs-all paradigm to classify the data into 4 different
classes(Orientation: 0, 90, 180, 270).

In this algorithm, we are classifying the data on a feature value such that the
error should be minimum. For this, we are calculating entropy to decide which
feature value classify data in the best possible way. After data classification,
we are calculating the error for misclassified data and adjusting the weights such
that weights of misclassified data should be more. In the next iteration,
adaboost try to classify data with bigger weights. This process repeats a
number of times and we create a number of weak classifiers.
So to classify test data, we check majority votes of weak classifiers to
decide label of unlabeled test data.
'''
def weighted_entropy(x, weights):
    # x is an array of 0's and 1's
    # weights is an array having float values corresponding to each value in x
    p0 = np.sum(weights[x == 0])
    p1 = np.sum(weights[x == 1])

    entropy = 0
    entropy -= p0*np.log(p0) if p0 != 0 else 0
    entropy -= p1*np.log(p1) if p1 != 0 else 0
    return entropy


def hypotheses_predict(hypothesis, X):
    n_examples = np.size(X, 0)
    prediction = np.zeros(n_examples, dtype=int) * hypothesis.labels[LEFT]
    prediction[X[:, hypothesis.attribute] >
               hypothesis.value] = hypothesis.labels[RIGHT]
    #print(prediction)
    return prediction


def compute_information_gain(X, y, attribute, value):
    current_entropy = entropy(y)

    split_condition = X[:, attribute] > attribute
    left_y, right_y = y[split_condition], y[~split_condition]
    return information_gain(right_y, left_y, current_entropy)



def build_hypothesis(X, y):
    n_examples, n_attributes = X.shape
    gains = [[attribute, value, compute_information_gain(X, y, attribute, value)] \
            for attribute in np.random.choice(range(n_attributes), size = 50, replace=False) \
            for value in np.random.choice(np.unique(X[:, attribute]), size = 50, replace=False)]

    attribute, value, gain = max(gains, key=lambda x: x[2])

    right_counts = Counter(y[X[:, attribute] > value])
    if right_counts:
        [(right_label, count)] = right_counts.most_common(1)
        left_label = 1 - right_label
    else:
        left_counts = Counter(y[X[:, attribute] <= value])
        [(left_label, count)] = left_counts.most_common(1)
        right_label = 1-left_label

    leaf_values = [left_label, right_label]

    hypothesis = Hypothesis(attribute=attribute,
                            value=value, labels=leaf_values)

    return hypothesis


class AdaboostClassifier:
    def __init__(self, n_hypotheses):
        self.n_hypotheses = n_hypotheses
        self.hypotheses = []
        self.decision_weights = []

    def fit(self, X, y):
        n_examples = np.size(X, 0)
        weights = np.array([1.0/n_examples for _ in range(n_examples)])
        for k in trange(self.n_hypotheses):
            indices = np.random.choice(range(n_examples), size=n_examples, replace=True, p=weights)

            weighted_X, weighted_y = X[indices], y[indices]
            hypothesis = build_hypothesis(weighted_X, weighted_y)


            prediction = hypotheses_predict(hypothesis, X)
            error = np.sum(weights[prediction != y])


            error= np.clip(error, a_min=1/(2*n_examples), a_max=1-1/(2*n_examples))
            update = error/(1-error)

            weights[prediction == y] *= update

            weights /= weights.sum()


            decision_weight = np.log(1-error)/(error)

            self.hypotheses.append(hypothesis)
            self.decision_weights.append(decision_weight)

    def predict(self, X):
        n_examples = np.size(X, 0)
        predictions = np.zeros((n_examples, 2))
        for h, z in zip(self.hypotheses, self.decision_weights):

            prediction = hypotheses_predict(h, X)
            for i in range(n_examples):
                if(prediction[i] ==  1):
                    predictions[i,0] = 1
                    predictions[i,1] = predictions[i,1] + 1
        return predictions


class_label = ["0","90","180","270"]


def run_adaboost(train_or_test, test_file, model_file):
    predicted_labels = []
    if(train_or_test == "test"):
        file = open(model_file,"rb")
        file.seek(0)
        pk_data = pickle.load(file)

        test = read_data(test_file)

        models = pk_data["models"]
        predictions = np.zeros(( np.size(test.pixel_values, 0), 2))
        count = 0
        for model in models:
            tmp_predictions = model.predict(test.pixel_values)
            for index in range(0, np.size(test.pixel_values,0)):
                if(tmp_predictions[index,0] == 1):
                    if(tmp_predictions[index][1] > predictions[index][1]):
                        predictions[index][0] = class_label[count]
                        predictions[index][1] = tmp_predictions[index][1]
            count += 1

        correct = 0

        for index in range(test.size):
            predicted_labels.append(int(predictions[index][0]))
            if(str(int(predictions[index][0])) == test.orientations[index]):
                correct += 1

        return list(zip(test.image_names, predicted_labels))
    else:
        train = read_data("train-data.txt")
        n_train = np.size(train.pixel_values, 0)
        models = []

        X = train.pixel_values
        for label in class_label:
            y = np.zeros(n_train, dtype=int)
            y[np.array(train.orientations) == label] = 1
            model = AdaboostClassifier(n_hypotheses=200)
            model.fit(X, y)
            models.append(model)

        adaboost_file = open(model_file,"wb")
        pickle.dump({
            "models": models
        },adaboost_file)
        adaboost_file.close()
