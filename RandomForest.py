from id3 import *
import csv
from imdbDataSet import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def random_forest(x_train, y_train, x_test, y_test, fv_skip_top, fv_length, tree_number, tree_fv_length, min_ig):

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
        id3_tree = ID3(features=encoded_tree_feature_vectors[i], min_ig=min_ig)
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
    sl_random_forest = RandomForestClassifier(n_estimators=tree_number)
    sl_random_forest.fit(train_examples,np.array(y_train))
    sklearn_random_forest_pred = sl_random_forest.predict(test_examples)


    return y_test, majority_outcome, sklearn_random_forest_pred

#Ορισμός υπερπαραμέτρων
sys.setrecursionlimit(3000)

train_step = np.linspace(5000,25000,15,dtype=int)
epoch_step = np.linspace(5,50,15,dtype=int)
tree_number = 23
fv_skip_top = 50
fv_length = 5000
tree_fv_length = 300
min_ig_for_split = 0.05 #used to determine the stopping point for id3 tree splits
vocab_size = 15000
max_sequence_length = 250

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


#create the rnn with gru layers sequential model
imdb_rnn_with_gru = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, output_dim = 32, input_length = max_sequence_length), #embedding layer
    tf.keras.layers.GRU(units=32),#gru layer 1
    tf.keras.layers.Dense(units=1, activation='sigmoid')  #output layer, output = probability of classification in the positive category
])

#compile the model
imdb_rnn_with_gru.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])


for step in range(len(train_step)):
    x_train = x_train_raw[:train_step[step]]
    y_train = y_train_raw[:train_step[step]]

    #training data tests
    y_test_train, y_pred_train_our, y_pred_train_sklearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length, min_ig=min_ig_for_split)
    
    print("Data for training tests was gathered at step -> " + str(train_step[step]))
    #testing data tests
    y_test_test, y_pred_test_our, y_pred_test_sklearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_test_raw, y_test=y_test_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length, min_ig=min_ig_for_split)
    print("Data for testing tests was gathered at step -> " + str(train_step[step]))

    #pad data sequences to fit into a specific size
    rnn_train_data = pad_sequences(sequences=x_train, maxlen=max_sequence_length)
    rnn_test_data = pad_sequences(sequences=x_test_raw, maxlen=max_sequence_length)


    #train the rnn model, we use a constant number of epochs to compare with the other implementations, the only variable here is the training data 
    imdb_rnn_with_gru.fit(rnn_train_data, y_train, epochs=10, batch_size=1024, validation_split=0.2)

    #obtain rnn model's predictions, if the probability is above 0.5 we consider it a positive classification, negative otherwise.
    #on training predictions
    y_pred_train_rnn = (imdb_rnn_with_gru.predict(rnn_train_data) > 0.5).astype("int32")
    #on testing predictions
    y_pred_test_rnn = (imdb_rnn_with_gru.predict(rnn_test_data) > 0.5).astype("int32")

    for metric, func in metrics.items():
        #calculate metrics for our implementation
        our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our))
        our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our))
        print('Calculated Our Metrics')


        #calculate metrics for sklearn implementation
        sklearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_sklearn))
        sklearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_sklearn))
        print('Calculated SKLearn Metrics')


        #calculate metrics for the sequential rnn implementation 
        rnn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_rnn))
        rnn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_rnn))
        print('Calculated RNN Metrics')

print("Data collection for all training steps has been completed!\n")
print('Gathering RNN Loss Data...\n')
#gather loss data from rnn, increasing epochs with constant train and test data for each loop 
#compile the model again
imdb_rnn_with_gru.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

rnn_train_data2 = pad_sequences(sequences=x_train_raw, maxlen=max_sequence_length)
rnn_test_data2 = pad_sequences(sequences=x_test_raw, maxlen=max_sequence_length)
for step in epoch_step:
    rnn_history = imdb_rnn_with_gru.fit(rnn_train_data2, y_train_raw, epochs=step, batch_size=1024, validation_split=0.2)
print('RNN Loss Data has been colected!\n')


print("Writing data into csv files...\n\n")
#write data results into the respective csv files
file_names = ['RandomForest_Accuracy.csv'
'RandomForest_Precision.csv'
'RandomForest_Recall.csv'
'RandomForest_F1.csv']
rnn_loss_file = 'RandomForest_rnn_Loss.csv'

#write the files with accuracy,precision,recall and f1 scores 
for file in file_names:
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for result in results.keys():
            writer.writerows(result + ' implementation')
            writer.writerows(['Training Data Step'] + [str(step) for step in train_step])
            writer.writerows(['Training Scores'] + [str(value) for value in results[result]['accuracy']['train']])
            writer.writerows(['Testing Scores'] + [str(value) for value in results[result]['accuracy']['test']])
            writer.writerows('\n')

#write the file with rnn losses
with open(rnn_loss_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows('RNN Losses based on epochs')
    writer.writerows(['Epochs Step'] + [str(step) for step in epoch_step])
    writer.writerows(['Training Loss'] + [str(loss) for loss in rnn_history.history['loss']])
    writer.writerows(['Validation Loss'] + [str(val_loss) for val_loss in rnn_history.history['val_loss']])

#plot the results
for metric in metrics.keys():
#plot the accuracy, precision, recall and f1 curves
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #subplot for our implementation
    axs[0].plot(train_step, our_results[metric]['train'], color='green', label='training data')
    axs[0].plot(train_step, our_results[metric]['test'], color='red', label='testing data')
    axs[0].set_title('Our Random Forest (Constant 11 trees)')
    axs[0].set_xlabel('Number of Training Data')
    axs[0].set_ylabel(f'Percent {metric.capitalize()}')
    axs[0].legend()

    #subplot for scikit-learn implementation
    axs[1].plot(train_step, sklearn_results[metric]['train'], color='green', label='training data')
    axs[1].plot(train_step, sklearn_results[metric]['test'], color='red', label='testing data')
    axs[1].set_title('Scikit-learn Random Forest (Constant 11 trees)')
    axs[1].set_xlabel('Number of Training Data')
    axs[1].set_ylabel(f'Percent {metric.capitalize()}')
    axs[1].legend()

    #subplot for rnn implementation
    axs[2].plot(train_step, rnn_results[metric]['train'], color='green', label='training data')
    axs[2].plot(train_step, rnn_results[metric]['test'], color='red', label='testing data')
    axs[2].set_title('Keras Sequential rnn (Constant 30 epochs)')
    axs[2].set_xlabel('Number of Training Data')
    axs[2].set_ylabel(f'Percent {metric.capitalize()}')
    axs[2].legend()

    #display the figure with the subplots
    plt.tight_layout()
    plt.show()

    #plot the rnn loss based on epochs
    plt.plot(epoch_step,rnn_history.history['loss'], color='blue', label='training data')
    plt.plot(epoch_step,rnn_history.history['val_loss'], color='orange', label='validation data')
    plt.set_title('Keras Sequential rnn (Constant all of the train-test data)')
    plt.set_xlabel('Epochs')
    plt.set_ylabel('Percent Loss')
    plt.legend()
    plt.show()
