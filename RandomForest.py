from id3 import *
from imdbDataSet import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def random_forest(x_train, y_train, x_test, y_test, fv_skip_top, fv_length, tree_number, tree_fv_length, min_ig, majority_for_split):

    #create encoded feature vector (index of words)
    encoded_feature_vector = imdb.getFeatureVector(skip_top=fv_skip_top, num_words=fv_length)



    #each tree has a feature vector smaller than the general feature vector
    #choose each feature at random
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
        id3_tree = ID3(features=encoded_tree_feature_vectors[i], min_ig=min_ig, majority_percentage=majority_for_split)
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
    sklearn_random_forest = RandomForestClassifier(n_estimators=tree_number)
    sklearn_random_forest.fit(train_examples,np.array(y_train))
    sklearn_random_forest_pred = sklearn_random_forest.predict(test_examples)


    return y_test, majority_outcome, sklearn_random_forest_pred

#Ορισμός υπερπαραμέτρων
sys.setrecursionlimit(3000)

train_step = np.linspace(5000,25000,10,dtype=int)
tree_number = 15
fv_skip_top = 45
fv_length = 600
tree_fv_length = 450
min_ig_for_split = 0.02 #used to determine the stopping point for id3 tree splits
majority_for_split = 0.85 #used to determine an early stopping point for id3 tree splits 
vocab_size = fv_length
max_sequence_length = 512

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
sklearn_results = {metric: {'train': [], 'test': []} for metric in metrics}
rnn_results = {metric: {'train': [], 'test': []} for metric in metrics}

results = {
    'our' : our_results,
    'sklearn' : sklearn_results,
    'rnn' : rnn_results
}


for step in range(len(train_step)):
    x_train = x_train_raw[:train_step[step]]
    y_train = y_train_raw[:train_step[step]]

    #training data tests
    y_test_train, y_pred_train_our, y_pred_train_sklearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length, min_ig=min_ig_for_split, majority_for_split=majority_for_split)
    
    print("Data for training tests was gathered at step -> " + str(train_step[step]))
    #testing data tests
    y_test_test, y_pred_test_our, y_pred_test_sklearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_test_raw, y_test=y_test_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length, min_ig=min_ig_for_split, majority_for_split=majority_for_split)
    print("Data for testing tests was gathered at step -> " + str(train_step[step]))

    #pad data sequences to fit into a specific size
    rnn_train_data = pad_sequences(sequences=x_train, maxlen=max_sequence_length)
    rnn_test_data = pad_sequences(sequences=x_test_raw, maxlen=max_sequence_length)

    #create the rnn with gru layers sequential model
    imdb_rnn_with_gru = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, output_dim = 32, input_length = max_sequence_length), #embedding layer
        tf.keras.layers.GRU(units=32, dropout=0.2, return_sequences = False),#gru layer 1
        tf.keras.layers.Dropout(rate=0.5),#dropout on last layer before output
        tf.keras.layers.Dense(units=1, activation='sigmoid')  #output layer, output = probability of classification in the positive category
    ])
    #compile the model
    imdb_rnn_with_gru.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])


    #train the rnn model, we use a constant number of epochs to compare with the other implementations, the only variable here is the training data 
    imdb_rnn_with_gru.fit(rnn_train_data, y_train, epochs=15, batch_size=1024, validation_data=(rnn_test_data, y_test_raw))

    #obtain rnn model's predictions, if the probability is above 0.5 we consider it a positive classification, negative otherwise.
    #on training predictions
    y_pred_train_rnn = (imdb_rnn_with_gru.predict(rnn_train_data) > 0.5).astype("int32")
    #on testing predictions
    y_pred_test_rnn = (imdb_rnn_with_gru.predict(rnn_test_data) > 0.5).astype("int32")

    for metric, func in metrics.items():
        #calculate metrics for our implementation
        our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our))
        our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our))
        print('Calculated Our ' + metric)
        templist_train = [str(val) for val in our_results[metric]['train']]
        templist_test = [str(val) for val in our_results[metric]['test']]
        print('Train: ')
        print(templist_train)
        print('Test: ')
        print(templist_test)

        #calculate metrics for sklearn implementation
        sklearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_sklearn))
        sklearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_sklearn))
        print('Calculated SKLearn ' + metric)
        templist_train = [str(val) for val in sklearn_results[metric]['train']]
        templist_test = [str(val) for val in sklearn_results[metric]['test']]
        print('Train: ')
        print(templist_train)
        print('Test: ')
        print(templist_test)

        #calculate metrics for the sequential rnn implementation 
        rnn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_rnn))
        rnn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_rnn))
        print('Calculated RNN ' + metric)
        templist_train = [str(val) for val in rnn_results[metric]['train']]
        templist_test = [str(val) for val in rnn_results[metric]['test']]
        print('Train: ')
        print(templist_train)
        print('Test: ')
        print(templist_test)

print("Data collection for all training steps has been completed!\n")


print('Gathering RNN Loss Data...\n')
#gather loss data from rnn, increasing epochs with constant train and test data for each loop 


#create the rnn with gru layers sequential model again
imdb_rnn_with_gru = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, output_dim = 32, input_length = max_sequence_length), #embedding layer
    tf.keras.layers.GRU(units=32, dropout=0.2, return_sequences = False),#gru layer 1
    tf.keras.layers.Dropout(rate=0.5),#dropout on last layer before output
    tf.keras.layers.Dense(units=1, activation='sigmoid')  #output layer, output = probability of classification in the positive category
])

#compile the model again
imdb_rnn_with_gru.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])

rnn_train_data2 = pad_sequences(sequences=x_train_raw, maxlen=max_sequence_length)
rnn_test_data2 = pad_sequences(sequences=x_test_raw, maxlen=max_sequence_length)
#run the fit method again for the last step in epoch_step
rnn_history = imdb_rnn_with_gru.fit(rnn_train_data2, y_train_raw, epochs=25, batch_size=1024, validation_data=(rnn_test_data, y_test_raw))

print('RNN Loss Data has been colected!\n')

#write results for accuracy,precision,recall,f1 scores.
print("Writing results...\n\n")
for metric in metrics.keys():
    print( metric.capitalize() + ' Results')
    train_step_list = ['Train Step: '] 
    train_step_list.extend([str(step) for step in train_step])
    print(train_step_list)
    for result in results.keys():
        print(result + ' implementation')
        train_results_list = ['Train Results: ']
        test_results_list = ['Test Results: ']
        train_results_list.extend([str(val) for val in results[result][metric]['train']])
        test_results_list.extend([str(val) for val in results[result][metric]['test']])
        print(train_results_list)
        print(test_results_list)


#write results for rnn losses
print('\n\nRNN with GRU Loss Data')
epochs_list = ['Epochs: ']
losses_list = ['Loss: ']
val_losses_list = ['Validation Loss: ']
epochs_list.extend([str(epoch) for epoch in np.linspace(1,25,25,dtype=int)])
losses_list.extend([str(loss) for loss in rnn_history.history['loss']])
val_losses_list.extend([str(loss) for loss in rnn_history.history['val_loss']])
print(epochs_list)
print(losses_list)
print(val_losses_list)

#plot the results
for metric in metrics.keys():
#plot the accuracy, precision, recall and f1 curves
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #subplot for our implementation
    axs[0].plot(train_step, our_results[metric]['train'], color='green', label='training data')
    axs[0].plot(train_step, our_results[metric]['test'], color='red', label='testing data')
    axs[0].set_title('Our Random Forest (Constant 15 trees)')
    axs[0].set_xlabel('Number of Training Data')
    axs[0].set_ylabel(f'Percent {metric.capitalize()}')
    axs[0].legend()

    #subplot for scikit-learn implementation
    axs[1].plot(train_step, sklearn_results[metric]['train'], color='green', label='training data')
    axs[1].plot(train_step, sklearn_results[metric]['test'], color='red', label='testing data')
    axs[1].set_title('Scikit-learn Random Forest (Constant 15 trees)')
    axs[1].set_xlabel('Number of Training Data')
    axs[1].set_ylabel(f'Percent {metric.capitalize()}')
    axs[1].legend()

    #subplot for rnn implementation
    axs[2].plot(train_step, rnn_results[metric]['train'], color='green', label='training data')
    axs[2].plot(train_step, rnn_results[metric]['test'], color='red', label='testing data')
    axs[2].set_title('Keras Sequential rnn (Constant 15 epochs)')
    axs[2].set_xlabel('Number of Training Data')
    axs[2].set_ylabel(f'Percent {metric.capitalize()}')
    axs[2].legend()

    #display the figure with the subplots
    plt.tight_layout()
    plt.show()

#plot the rnn loss based on epochs
plt.plot(np.linspace(1,25,25,dtype=int),rnn_history.history['loss'], color='blue', label='training data')
plt.plot(np.linspace(1,25,25,dtype=int),rnn_history.history['val_loss'], color='orange', label='validation data')
plt.title('Keras Sequential rnn (Constant all of the train-test data)')
plt.xlabel('Epochs')
plt.ylabel('Percent Loss')
plt.legend()
plt.tight_layout()
plt.show()