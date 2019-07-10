#!/usr/bin/env python3
from tree import *
from dataset import *

#Parameters of the model
num_trees = 1
#test_data_size = 0.3           #Percentage of total data
max_tree_depth = 2

'''
Random forest classifier relies on the technique of creating many decision trees to avoid over-fitting. The idea is that each of these trees will capture some information about the dataset and when combined together they will generalize better than a single decision tree.

In our implementation of RandomForest classifier, we have made the following design decisions:
For selecting the best split criteria, we have maximized information gain over a randomly selected list of features and thresholds.
No further splitting is done if 3 or less than 3 examples are left at the node.
A subset of data is used for creating each of the decision trees.

A detailed explanation is provided in the report
'''
def random_forest(train_or_test, file_name, model_file):

    #Read test dataset
    data = read_data(file_name)
    test_data = data[2]
    test_labels = data[1]

    if(train_or_test == "test"):
        dict_ = load_json(model_file)
        a = []
        i = 0
        for key in dict_:
            a.append(Tree(max_tree_depth))
            a[i].assign_json(dict_[key])
            i += 1
        predicted_labels = predict(test_data, a)
        return list(zip(data.image_names, predicted_labels))
    else:
        #Read the data from the files
        data = read_data("train-data.txt")
        train_labels = data[1]
        train_data = data[2]
        model_dict = {}
        a = [Tree(max_tree_depth) for i in range(0,num_trees)]
        for i in range(0,num_trees):
            print("Creating tree: ", i)
            r_train_data, r_train_labels = gen_rand_train_data(train_data, train_labels, len(train_data)//50)
            print("Length of the data: ", len(r_train_labels))
            a[i].build(train_data, train_labels)
            model_dict[i] = a[i].create_dict()
        save_json(model_dict, model_file)
        print("Model trained")
