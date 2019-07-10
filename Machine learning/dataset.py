from random import sample, shuffle
from collections import namedtuple
import numpy as np

Dataset = namedtuple("Dataset", ["image_names", "orientations", "pixel_values", "size"])

def read_data(file_path):
    with open(file_path, "r") as in_file:
        rows = [line.split() for line in in_file]
        image_names = [row[0] for row in rows]
        orientations = [row[1] for row in rows]
        pixel_values = [list(map(int, row[2:])) for row in rows]

        return Dataset(image_names, orientations, np.array(pixel_values), len(image_names))

def subset_data(data, indices):
    image_names, orientations, pixel_values, _ = data
    return Dataset(image_names=[image_names[i] for i in indices],
                   orientations=[orientations[i] for i in indices],
                   pixel_values=np.array([pixel_values[i] for i in indices]),
                   size=len(indices))

def train_test_split(data, fraction=.7):
    N = data.size
    indices = list(range(N))
    shuffle(indices)

    train_indices = sample(indices, int(fraction*N))
    train_data = subset_data(data, train_indices)

    test_indices = frozenset(indices) - frozenset(train_indices)
    test_data = subset_data(data, test_indices)

    return train_data, test_data



# Divide the data randomly in training and testing set as per the size provided
# Parameters:
#   test_size: Size of the test set
#   random : Used for selecting data randomly or sequentially
def gen_rand_test_data(data, labels, test_size=0, random = True):

    #arrays for saving the testing and training data
    train_data = np.empty((data.shape[0]- test_size, data.shape[1]))
    train_labels = np.empty((data.shape[0]- test_size, 1))


    test_data = np.empty((test_size, data.shape[1]))
    test_labels = np.empty((test_size, 1))

    if(test_size == 0):
        return data, labels, test_data, test_labels
    data_key = list(range(0,data.shape[0]))
    if(random == False):
        print("Test data selected sequentially")
        test_key_random = data_key[:test_size]
    else:
        test_key_random = np.random.choice(data_key, test_size, replace=False)
        test_key_random.sort()

    #Iterations count for adding the data in two different groups
    train_index = 0
    test_index = 0
    for i in data_key:
        if(i not in test_key_random):
            train_data[train_index] = data[i]
            train_labels[train_index] = labels[i]
            train_index += 1
        else:
            test_data[test_index] =  data[i]
            test_labels[test_index] = labels[i]
            test_index += 1

    return train_data, train_labels, test_data, test_labels
