from id3 import *
from imdbDataSet import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import random
import math
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def random_forest(x_train, y_train, x_test, y_test, fv_skip_top, fv_length, tree_number, tree_fv_length, max_depth):

    #create encoded feature vector (index of words)
    encoded_feature_vector = imdb.getFeatureVector(skip_top=fv_skip_top, num_words=fv_length)



    #each tree has a feature vector smaller than the original
    #choose each attribute at random
    encoded_tree_feature_vectors = []
    for i in range(tree_number):
        encoded_tree_feature_vectors.append(random.sample(encoded_feature_vector, tree_fv_length))
        encoded_tree_feature_vectors[i].sort()



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
        id3_tree = ID3(features=encoded_tree_feature_vectors[i], max_depth=max_depth)
        id3_tree.fit(train_examples, np.array(y_train))
        forest.append(id3_tree)



    #collect test example predictions from the forest
    all_tree_predictions = np.zeros((tree_number, len(y_test)))
    for i in range(tree_number):
        all_tree_predictions[i,:] = forest[i].predict(test_examples)


        
    #organize each tree's prediction to be calculated
    tree_votes = np.sum(all_tree_predictions, axis=0)



    #implement majority vote for each example
    majority_outcome = np.zeros(len(test_examples))
    #majority predicted positive outcome (1) when majority vote is greater than 0.5
    #majority predicted negative outcome (0) when majority vote is lesser or equal than 0.5
    majority_outcome = np.where(tree_votes / len(forest) > 0.5, 1, 0)



    #implement sklearn's random forest for comparison
    sl_random_forest = RandomForestClassifier(n_estimators=tree_number, max_features=max_depth)
    sl_random_forest.fit(train_examples,np.array(y_train))
    sl_random_forest_pred = sl_random_forest.predict(test_examples)

    return y_test, majority_outcome, sl_random_forest_pred

#Ορισμός υπερπαραμέτρων
sys.setrecursionlimit(3000)

train_step = np.linspace(5000,25000,15,dtype=int)
tree_number = 7
fv_skip_top = 80
fv_length = 200
tree_fv_length = 75
id3_max_depth = math.ceil(math.sqrt(tree_fv_length))#this value must be lesser or equal to tree_fv_length


#obtain imdb data 
imdb = IMDB()
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.getTrainingData(skip_top=fv_skip_top, num_words=fv_length)

#initialize metrics
metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
}

our_results = {metric: {'train': [], 'test': []} for metric in metrics}
scilearn_results = {metric: {'train': [], 'test': []} for metric in metrics}

for step in train_step:
    x_train = x_train_raw[:step]
    y_train = y_train_raw[:step]

    #training data tests
    y_test_train, y_pred_train_our, y_pred_train_scilearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_train_raw, y_test=y_train_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length, max_depth = id3_max_depth)
    
    print("Data for training tests was gathered at step -> " + str(step))
    #testing data tests
    y_test_test, y_pred_test_our, y_pred_test_scilearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_test_raw, y_test=y_test_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length, max_depth = id3_max_depth)
    print("Data for testing tests was gathered at step -> " + str(step))

    for metric, func in metrics.items():
        #calculate metrics for our implementation
        if metric == 'precision' or metric == 'recall':
            our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our, zero_division=1))
            our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our, zero_division=1))
        else:
            our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our))
            our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our))


        #calculate metrics for scilearn implementation
        if metric == 'precision' or metric == 'recall':
            scilearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_scilearn, zero_division=1))
            scilearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_scilearn, zero_division=1))
        else:
            scilearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_scilearn))
            scilearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_scilearn))

print("Data collection for all training steps has been completed!")
print("\n\n")
print("[Our Random Forest] Metric " + metric + " for training data is calculated at: " +  str(our_results[metric]['train']))
print("[Our Random Forest] Metric " + metric + " for testing data is calculated at: " +  str(our_results[metric]['test']))
print("\n\n")
print("[Scikit Learn] Metric " + metric + " for training data is calculated at: " +  str(scilearn_results[metric]['train']))
print("[Scikit Learn] Metric " + metric + " for testing data is calculated at: " +  str(scilearn_results[metric]['test']))
print("\n\n")


#plot the results
for metric in metrics.keys():
#plot the accuracy, precision, recall and f1 curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Subplot for our implementation
    axs[0].plot(train_step, our_results[metric]['train'], color='green', label='training data')
    axs[0].plot(train_step, our_results[metric]['test'], color='red', label='testing data')
    axs[0].set_title('Our Random Forest')
    axs[0].set_xlabel('Number of Training Data')
    axs[0].set_ylabel(f'Percent {metric.capitalize()}')
    axs[0].legend()

    # Subplot for scikit-learn implementation
    axs[1].plot(train_step, scilearn_results[metric]['train'], color='green', label='training data')
    axs[1].plot(train_step, scilearn_results[metric]['test'], color='red', label='testing data')
    axs[1].set_title('Scikit-learn Random Forest')
    axs[1].set_xlabel('Number of Training Data')
    axs[1].set_ylabel(f'Percent {metric.capitalize()}')
    axs[1].legend()

    # Display the figure with the subplots
    plt.tight_layout()
    plt.show()


