#!/usr/bin/env python3

import sys
from randomForest import *
from knn import *
from adaboost_mk_2 import *

test_or_train = sys.argv[1]
file_name = sys.argv[2]
model_file = sys.argv[3]
model_name = sys.argv[4]

if(model_name == 'nearest'):
    #print(knn_predict(file_name, model_name))
    knn_data = knn_predict(test_or_train, file_name, model_file)
    with open('output.txt', 'w') as f:
        f.write('\n'.join('%s %s' % item for item in knn_data))

if(model_name == 'adaboost'):
    adaboost_data = run_adaboost(test_or_train, file_name, model_file)
    with open('output.txt', 'w') as f:
        f.write('\n'.join('%s %s' % item for item in adaboost_data))

if(model_name == 'forest'):
    #print(random_forest(test_or_train, file_name, model_file))
    forest_data = random_forest(test_or_train, file_name, model_file)
    with open("output.txt", 'w') as f:
        f.write('\n'.join('%s %s' % item for item in forest_data))

if(model_name == 'best'):
    forest_data = random_forest(test_or_train, file_name, model_file)
    with open("output.txt", 'w') as f:
        f.write('\n'.join('%s %s' % item for item in forest_data))
