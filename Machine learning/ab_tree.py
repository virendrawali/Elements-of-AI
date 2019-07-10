import numpy as np
from math import inf, log
from collections import namedtuple, Counter
import json
from collections import defaultdict

possible_classes = ['0', '90', '180', '270']


class Tree:

    leaf_node_threshold = 3

    def __init__(self, depth):
        #self.left_node = None
        #self.right_node = None
        self.depth = depth
        self.feature = -1
        self.threshold = None
        self.entropy = None
        #self.n_node_samples = None
        #self.is_leaf_node = False
        #self.leaf_results = {}
        self.alpha = None
        self.label_class = None
        #self.leaf_results = defaultdict(default_factory=lambda :0)

        '''
        Taken from sklearn implementation
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        '''

    def test_leaf_node(self, labels):

        #If all examples in the node are same then, marks as leaf node
        count = Counter(labels)
        counter = 0
        for key in count:
            counter += 1
        if(counter <= 1):
            return True

        #if depth is 0 or less than 3 examples are present mark as leaf node
        if(self.depth == 0 or self.n_node_samples <= self.leaf_node_threshold ):
            return True
        else:
            return False

    def build(self,data, labels, weights):
        self.feature, self.threshold = find_best_split(data, labels, weights)
        right_index, right_data, right_labels, left_index, left_data, left_labels = split_data(data, labels, self.feature,self.threshold)
        #print("right data", right_index)
        #print("right label", right_labels)
        #print("left data", left_index)
        #print("left label", left_labels)
        return split_data(data, labels, self.feature, self.threshold)

    def traverse_tree(self):
        output = "Depth: {0}, # of nodes: {1}, is leaf_node: {2}".format(self.depth, self.n_node_samples, self.is_leaf_node)
        #print(output)
        output = "Threshold: {0}, Feature: {1}".format(self.threshold, self.feature)
        #print(output)
        if(self.is_leaf_node == True):
            print("Leaf node results: "),
            print(self.leaf_results)
            print("\n")
            return 0

        else:
            print("\n")

        self.right_node.traverse_tree()
        self.left_node.traverse_tree()
        return 0


    def check_input(self, data):
        '''
        returns the decision of the tree for the given input vector
        data: one single input example
        '''
        if(self.is_leaf_node):
            #print(self.leaf_results)
            #results = {}
            #for key in self.leaf_results:
            #    results[key] = self.leaf_results[key]*self.n_node_samples
            #print(results)
            return self.leaf_results
        #output = "Threshold: {0}, Feature: {1}".format(self.threshold, self.feature)

        if(data[self.feature] > self.threshold):
            return self.right_node.check_input(data)
        else:
            return self.left_node.check_input(data)


    # A method for saving object data to JSON file
    def create_dict(self):

        dict_ = {}
        dict_['depth'] = self.depth
        dict_['feature'] = self.feature
        dict_['threshold'] = self.threshold
        dict_['class'] = self.label_class
        dict_['alpha'] = self.alpha

        return dict_


    def assign_json(self, dict_):
        self.depth = int(dict_['depth'])
        self.feature = int(dict_['feature'])
        if dict_['threshold'] is not None:
            self.threshold = int(dict_['threshold'])
        else:
            self.threshold = dict_['threshold']
        self.is_leaf_node = bool(dict_['isLeafNode'])
        self.n_node_samples = int(dict_['numOfNodes'])
        self.leaf_results = dict_['leafResults']
        for key in self.leaf_results:
            self.leaf_results[key] = float(self.leaf_results[key])
        if(self.is_leaf_node):
            return 0
        else:
            self.left_node = Tree(self.depth - 1)
            self.right_node = Tree(self.depth - 1)

            self.left_node.assign_json(dict_['left_node'])
            self.right_node.assign_json(dict_['right_node'])


def save_json(dict_, filepath):
    # Creat json and save to file
    json_txt = json.dumps(dict_, indent=4)
    with open(filepath, 'w') as file:
        file.write(json_txt)

def load_json(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
    return data


def split_check_info_gain(data, labels, scig_weights, feature, threshold):
    right_node_labels = []
    right_node_labels_wt = []
    left_node_labels = []
    left_node_labels_wt = []
    for index in range(0, len(data)):
        if(data[index][feature] > threshold):
            right_node_labels.append(labels[index])
            right_node_labels_wt.append(scig_weights[index])
        else:
            #print("less than",data[index][feature])
            #print("less label", labels[index])
            left_node_labels.append(labels[index])
            left_node_labels_wt.append(scig_weights[index])
    #print("left_node_labels in split",left_node_labels)
    return right_node_labels, right_node_labels_wt, left_node_labels, left_node_labels_wt

def split_data(data, labels, feature, threshold):
    right_data_index = []
    right_node_data = []
    right_node_labels = []

    left_data_index = []
    left_node_data = []
    left_node_labels = []

    for index in range(0, len(data)):
        if(data[index][feature] > threshold):
            right_data_index.append(index+1)
            right_node_data.append(data[index])
            right_node_labels.append(labels[index])
        else:
            left_data_index.append(index+1)
            left_node_data.append(data[index])
            left_node_labels.append(labels[index])

    return right_data_index, right_node_data, right_node_labels, left_data_index,left_node_data, left_node_labels


def unique(list_data):
    unique_list = []
    for item in list_data:
        if(item not in unique_list):
            unique_list.append(item)
    return unique_list


def find_best_split(data, labels,weights):
    checked_values = {}
    max = -inf
    selected_index = -1
    selected_threshold = None

    current_node_entropy = entropy(labels,weights)
    #print("Current entropy",current_node_entropy)
    #print("Current node entropy: ", current_node_entropy)

    for index in np.random.choice(range(len(data[0])), size = 70):
        unique_col_vals = np.unique([row[index] for row in data])
        #print(unique_col_vals)

        checked_values[index] = {}

        for threshold in np.random.choice(unique_col_vals, size = 30):

            right_labels_bs,right_labels_wt,left_labels_bs,left_labels_wt = split_check_info_gain(data,labels,weights, index, threshold)
            split_info_gain = information_gain(right_labels_bs,right_labels_wt, left_labels_bs,left_labels_wt,current_node_entropy)
            checked_values[index][threshold] = split_info_gain
            if(split_info_gain > max):
                max = split_info_gain
                selected_index = index
                selected_threshold = threshold
    return np.asscalar(selected_index), np.asscalar(selected_threshold)

def entropy(ls,etp_weights):
    total = len(ls)
    class1_weight = 0
    class2_weight = 0
    ps = []
    for i in range(0,len(ls)):
        if(ls[i] == 1):
            class1_weight+=etp_weights[i]
        else:
            class2_weight+=etp_weights[i]
    ps.append(class1_weight/total)
    ps.append(class2_weight/total)
    return sum(-p * log(p) if p != 0 else 0 for p in ps)

def information_gain( right_node_labels,right_node_labels_wt, left_node_labels,left_node_labels_wt, current_node_entropy):
    n_left_wt = len(left_node_labels_wt)
    n_right_wt = len(right_node_labels_wt)
    total = n_left_wt + n_right_wt
    if(n_left_wt == 0):
        left_entropy = 0
    else:
        left_entropy = entropy(left_node_labels,left_node_labels_wt)
    if(n_right_wt == 0):
        right_entropy = 0
    else:
        right_entropy = entropy(right_node_labels,right_node_labels_wt)
    #print("Entropies (left, right): ",left_entropy, right_entropy)
    return current_node_entropy - (n_left_wt*left_entropy + n_right_wt*right_entropy)/total

def predict(data, tree_array):
    
    final_predicted_labels = []


    for row in data:

        predicted_probabilities = {i:0 for i in possible_classes}
    
        for tree in tree_array:
            tree_prediction = tree.check_input(row)
         
            #print("Prediction of tree: ")
            #print(tree_prediction)
            
            max_val = 0
            for key in tree_prediction:
                if(tree_prediction[key] > max_val):
                    max_val = tree_prediction[key]
                    max_key = key
           
            predicted_probabilities[max_key] =  predicted_probabilities[max_key]  + 1
            #print(predicted_probabilities[max_key])
            #predicted_probabilities[key] += tree_prediction[key]
        print(predicted_probabilities)
        final_predicted_labels.append(max(predicted_probabilities, key=lambda key: predicted_probabilities[key]))
    
    return final_predicted_labels

def accuracy(predicted_labels, labels):
    total = len(labels)
    count = 0
    for index in range(0, len(predicted_labels)):
        if(predicted_labels[index] == labels[index]):
            count += 1

    return count/total

# Generate random data from the training set given the size of the data required
# Data will be used for creating the learning curves
# Size: size of the training required
def gen_rand_train_data(data, labels, size = 0):

    if(size == 0):
        return data, labels

    #data_key = list(range(0,data.shape[0]))
    data_key = list(range(0,len(labels)))
    list_random = np.random.choice(data_key, size, replace=False)
    list_random.sort()

    #random_data = np.empty((size, data.shape[1]))
    #random_labels = np.empty((size, 1))

    random_data = []
    random_labels = []
    index=0
    for i in list_random:
        random_data.append(data[i])
        random_labels.append(labels[i])
        index += 1
    return (random_data, random_labels)
