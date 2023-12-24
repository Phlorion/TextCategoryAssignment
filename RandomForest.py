from id3 import *
from imdbDataSet import *
import numpy as np
import random

# 
# 
#   !!!FIX ID3 FIT METHOD PROBLEM (2-D Arrays)!!!
# 
# 


#Ορισμός υπερπαραμέτρων
train_n = 1000
test_n = 100
tree_number = 5
fv_skip_top = 200
fv_num_words = 1000

#obtain imdb data 
imdb = IMDB()
(x_train_raw, y_train), (x_test_raw, y_test) = imdb.getTrainingData(skip_top=fv_skip_top, num_words=fv_num_words)

#use train_n amount of training examples
x_train = x_train_raw[:train_n]
print("Training Examples Number: ", str(len(x_train)))
y_train = y_train[:train_n]


#use test_n amount of testing examples
x_test = x_test_raw[:test_n]
print("Testing Examples Number: ", str(len(x_test)))
y_test = y_test[:test_n]


#create encoded feature vector (index of words)
encoded_feature_vector = imdb.getFeatureVector(skip_top=fv_skip_top, num_words=fv_num_words)

#create 0-1 feature vector for each training example
train_examples = np.zeros_like(x_train)
for i in range(len(x_train)):
    example = np.zeros(len(encoded_feature_vector))
    for w in x_train[i]:  
        if(w in encoded_feature_vector):
            example[encoded_feature_vector.index(w)] = 1
    train_examples[i] = example

print(train_examples[:3])

#create 0-1 feature vector for each test example
test_examples = np.zeros_like(x_test)
for i in range(len(x_test)):
    example = np.zeros(len(encoded_feature_vector))
    for w in x_test[i]:  
        if(w in encoded_feature_vector):
            example[encoded_feature_vector.index(w)] = 1
    test_examples[i] = example

#create ID3 trees for the ensamble
trees = []
for i in range(tree_number):
    id3_tree = ID3(features=encoded_feature_vector)
    id3_tree.fit(x = train_examples, y = y_train)
    trees.append(id3_tree)


#collect test example predictions from all trees
all_tree_predictions = np.zeros(len(test_examples))
for t in trees:
    for i in range(len(test_examples)):
        all_tree_predictions[i] = all_tree_predictions[i] + t.predict(test_examples[i])

#implement majority vote for each example
majority_outcome = np.zeros(len(test_examples))
for i in range(len(all_tree_predictions)):
    #majority predicted positive outcome (1)
    if(all_tree_predictions[i] / (len(trees)* 1.0) > 0.5):
        majority_outcome[i] = 1

    #majority predicted negative outcome (0)
    elif(all_tree_predictions[i] / (len(trees)* 1.0) < 0.5):
        majority_outcome[i] = 0

    #majority does not exists, both outcomes equally predicted -> choose randomly 0 or 1
    else:
        majority_outcome[i] = random.randint(0,1)

#calculate errors vector
#for each example we use 0 if the predection is correct, 1 if the prediction is erroneous
errors = np.zeros(len(x_test))
for i in range(x_test):
    #absolute value defines the distance of two numbers, if the distance is not 0 then 
    #the prediction is different than the actual value of y_test, therefor an error
    errors[i] = abs(y_test[i] - majority_outcome[i]) 

#show results for the first n test examples
n = 200
print(y_test[:n])
print(majority_outcome[:n])

#show error as percentage 
print(sum(errors)/(len(errors)*1.0))
