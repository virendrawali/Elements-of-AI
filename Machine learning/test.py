import numpy as np
import tree as rF
from dataset import *

def test_split_data():
    dataset = [[2.771244718,1.784783929,0],
    	[1.728571309,1.169761413,0],
    	[3.678319846,2.81281357,0],
    	[3.961043357,2.61995032,0],
    	[2.999208922,2.209014212,0],
    	[7.497545867,3.162953546,1],
    	[9.00220326,3.339047188,1],
    	[7.444542326,0.476683375,1],
    	[10.12493903,3.234550982,1],
    	[6.642287351,3.319983761,1]]

    dataset = np.asarray(dataset)
    train_data = np.delete(dataset, 2, axis=1)
    train_labels = np.delete(dataset, (0,1), axis=1)

    print(rF.entropy((train_labels.reshape(1,10)[0].tolist())))
    r_d,r_l,l_d,l_l = rF.split_data(train_data, train_labels, 0, 2)
    print(r_d,r_l,l_d,l_l)

#Dataset = namedtuple("Dataset", ["image_names", "orientations", "pixel_values", "size"])

test_split_data()
data = read_data("train-data.txt")
train_data = data[2][:100]
train_labels = data[1][:100]

a = rF.Tree(-1)
#dict_ = rF.load_json('random_forest.model')
#a.build(train_data, train_labels)
#a.build(train_data, train_labels)
#dict_ = a.create_dict()
filepath = 'random_forest.model'
#rF.save_json(dict_,filepath)
dict_ = rF.load_json(filepath)
a.assign_json(dict_)
a.traverse_tree()
a.traverse_tree()
print(a.check_input(train_data[:1][0]))
#a.save_json('random_forest.model')
