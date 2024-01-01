from id3 import *
from imdbDataSet import *
import numpy as np
import random
import sys


#Ορισμός υπερπαραμέτρων
sys.setrecursionlimit(3000)

train_n = 5000
test_n = 10000
tree_number = 10
fv_skip_top = 4000
fv_length = 300
tree_fv_length = 50

#obtain imdb data 
imdb = IMDB()
(x_train_raw, y_train), (x_test_raw, y_test) = imdb.getTrainingData(skip_top=fv_skip_top, num_words=fv_length)

#use train_n amount of training exampl
x_train = x_train_raw[:train_n]
print("Training Examples Number: ", str(len(x_train)))
y_train = y_train[:train_n]


#use test_n amount of testing examples
x_test = x_test_raw[:test_n]
print("Testing Examples Number: ", str(len(x_test)))
y_test = y_test[:test_n]


#create encoded feature vector (index of words)
encoded_feature_vector = imdb.getFeatureVector(skip_top=fv_skip_top, num_words=fv_length)
print("Encoded Feature Vector: ", encoded_feature_vector)

#each tree has a feature vector smaller than the original
#choose each attribute at random
encoded_tree_feature_vectors = []
for i in range(tree_number):
    encoded_tree_feature_vectors.append(random.sample(encoded_feature_vector, tree_fv_length))


#create 0-1 feature vector for each training example
train_examples = np.zeros_like(x_train)
for i in range(len(x_train)):
    example = np.zeros(len(encoded_feature_vector))
    for w in x_train[i]:  
        if(w in encoded_feature_vector):
            example[encoded_feature_vector.index(w)] = 1
    train_examples[i] = example

#reshape train examples to 2D array
train_examples = np.stack(train_examples)


#create 0-1 feature vector for each test example
test_examples = np.zeros_like(x_test)
for i in range(len(x_test)):
    example = np.zeros(len(encoded_feature_vector))
    for w in x_test[i]:  
        if(w in encoded_feature_vector):
            example[encoded_feature_vector.index(w)] = 1
    test_examples[i] = example
#reshape test examples to 2D array
test_examples = np.stack(test_examples)


#create ID3 forest for the ensamble
forest = []
for i in range(tree_number):
    id3_tree = ID3(features=encoded_tree_feature_vectors[i])
    id3_tree.fit(np.array(train_examples), np.array(y_train))
    forest.append(id3_tree)


#collect test example predictions from the forest
all_tree_predictions = np.zeros((tree_number, len(y_test)))
for i in range(tree_number):
    all_tree_predictions[i] = forest[i].predict(test_examples)

#organize each tree's prediction to be calculated
tree_votes = np.zeros(len(y_test))
for i in range(len(y_test)):
    for j in range(len(forest)):
        tree_votes[i] += all_tree_predictions[j][i]



#implement majority vote for each example
majority_outcome = np.zeros(len(test_examples))
for i in range(len(all_tree_predictions)):
    #majority predicted positive outcome (1)
    if(tree_votes[i] / (len(forest)* 1.0) > 0.5):
        majority_outcome[i] = 1

    #majority predicted negative outcome (0)
    elif(tree_votes[i] / (len(forest)* 1.0) < 0.5):
        majority_outcome[i] = 0

    #majority does not exists, both outcomes equally predicted -> choose randomly 0 or 1
    else:
        majority_outcome[i] = random.randint(0,1)

#calculate errors vector
#for each example we use 0 if the predection is correct, 1 if the prediction is erroneous
errors = np.zeros(len(x_test))
for i in range(len(x_test)):
    #absolute value defines the distance of two numbers, if the distance is not 0 then 
    #the prediction is different than the actual value of y_test, therefor an error
    errors[i] = abs(y_test[i] - majority_outcome[i]) 

#show results for the first n test examples
n = len(x_test)
print("Showing the first ", n, " expected answers:")
print(y_test[:n])
print("Showing the first ", n, " predicted answers:")
print(majority_outcome[:n])

#show error as percentage 
print("Percentage of error is: ", sum(errors)/(len(errors)*1.0))
