from id3 import *
from imdbDataSet import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def random_forest(x_train, y_train, x_test, y_test, fv_skip_top, fv_length, tree_number, tree_fv_length):
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
        id3_tree = ID3(features=encoded_tree_feature_vectors[i])
        id3_tree.fit(train_examples, np.array(y_train))
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

        #majority does not exist, both outcomes equally predicted -> choose randomly 0 or 1
        else:
            majority_outcome[i] = random.randint(0,1)

    sl_random_forest = RandomForestClassifier(n_estimators=tree_number)
    sl_random_forest.fit(train_examples,np.array(y_train))
    sl_random_forest_pred = sl_random_forest.predict(test_examples)
    
    return y_test, majority_outcome, sl_random_forest_pred

#Ορισμός υπερπαραμέτρων
sys.setrecursionlimit(3000)

train_step = np.linspace(1000,25000,30,dtype=int)
tree_number = 5
fv_skip_top = 150
fv_length = 100
tree_fv_length = 25


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
    y_test_train, y_pred_train_our, y_pred_train_scilearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_train_raw, y_test=y_train_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length)
    
    #testing data tests
    y_test_test, y_pred_test_our, y_pred_test_scilearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_test_raw, y_test=y_test_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length)
    
    for metric, func in metrics.items():
        #calculate metrics for our implementation
        if metric == 'precision':
            our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our, zero_division=1))
            our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our, zero_division=1))
        else:
            our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our))
            our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our))

        #calculate metrics for scilearn implementation
        if metric == 'precision':
            scilearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_scilearn, zero_division=1))
            scilearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_scilearn, zero_division=1))
        else:
            scilearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_scilearn))
            scilearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_scilearn))

# Plot results
for metric in metrics.keys():
#plot the presicion/recall plots differently than f1 and accuracy
    if metric == 'recall':
        continue
 
    
    if metric == 'precision':
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # Subplot for our implementation
        axs[0].plot(our_results['recall']['train'], our_results['precision']['train'], color='green', label='training data')
        axs[0].plot(our_results['recall']['test'], our_results['precision']['test'], color='red', label='testing data')
        axs[0].set_title('Our Random Forest')
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[0].legend()

        # Subplot for scikit-learn implementation
        axs[1].plot(scilearn_results['recall']['train'], scilearn_results['precision']['train'], color='green', label='training data')
        axs[1].plot(scilearn_results['recall']['test'], scilearn_results['precision']['test'], color='red', label='testing data')
        axs[1].set_title('Scikit-learn Random Forest')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].legend()
        
        # Display the figure with the subplots
        plt.tight_layout()
        plt.show()
    #for accuracy and f1 measure
    elif metric != 'precision':
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


