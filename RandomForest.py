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
    sl_random_forest_pred = sl_random_forest.predict(test_examples)


    return y_test, majority_outcome, sl_random_forest_pred

#Ορισμός υπερπαραμέτρων
sys.setrecursionlimit(3000)

train_step = np.linspace(5000,25000,15,dtype=int)
epoch_step = np.linspace(30,100,15,dtype=int)
tree_number = 9
fv_skip_top = 75
fv_length = 250
tree_fv_length = 80


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
mlp_results = {metric: {'train': [], 'test': []} for metric in metrics}

#create the mlp sequential model
imdb_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(12000, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid')  #binary classification
])
#compile the model
imdb_mlp.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])


for step in range(len(train_step)):
    x_train = x_train_raw[:train_step[step]]
    y_train = y_train_raw[:train_step[step]]

    #training data tests
    y_test_train, y_pred_train_our, y_pred_train_sklearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length)
    
    print("Data for training tests was gathered at step -> " + str(train_step[step]))
    #testing data tests
    y_test_test, y_pred_test_our, y_pred_test_sklearn = random_forest(x_train=x_train, y_train=y_train, x_test=x_test_raw, y_test=y_test_raw, fv_skip_top=fv_skip_top, fv_length=fv_length,tree_number=tree_number,tree_fv_length=tree_fv_length)
    print("Data for testing tests was gathered at step -> " + str(train_step[step]))

    #pad data sequences to fit into a specific size
    mlp_train_data = pad_sequences(x_train, value=0, padding='post', maxlen=512)
    mlp_test_data = pad_sequences(x_test_raw, value=0, padding='post', maxlen=512)



    #train the mlp model
    #validation data is the testing data used similarly by the random forests classifiers.
    imdb_mlp.fit(mlp_train_data, y_train, epochs=epoch_step[step], batch_size=512, validation_data=(mlp_test_data, y_test_raw))

    #obtain mlp model's predictions, if the probability is above 0.5 we consider it a positive classification, negative otherwise.
    #on training predictions
    y_pred_train_mlp = (imdb_mlp.predict(mlp_train_data) > 0.5).astype("int32")
    #on testing predictions
    y_pred_test_mlp = (imdb_mlp.predict(mlp_test_data) > 0.5).astype("int32")


    print("Data from MLP Sequential model was gathered at epoch -> " + str(epoch_step[step]))

    for metric, func in metrics.items():
        #calculate metrics for our implementation
        our_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_our))
        our_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_our))


        #calculate metrics for sklearn implementation
        sklearn_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_sklearn))
        sklearn_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_sklearn))


        #calculate metrics for the sequential mlp implementation 
        mlp_results[metric]['train'].append(func(y_true=y_test_train, y_pred=y_pred_train_mlp))
        mlp_results[metric]['test'].append(func(y_true=y_test_test, y_pred=y_pred_test_mlp))

print("Data collection for all training steps has been completed!")
print("\n\n")



#plot the results
for metric in metrics.keys():
#plot the accuracy, precision, recall and f1 curves
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #subplot for our implementation
    axs[0].plot(train_step, our_results[metric]['train'], color='green', label='training data')
    axs[0].plot(train_step, our_results[metric]['test'], color='red', label='testing data')
    axs[0].set_title('Our Random Forest')
    axs[0].set_xlabel('Number of Training Data')
    axs[0].set_ylabel(f'Percent {metric.capitalize()}')
    axs[0].legend()

    #subplot for scikit-learn implementation
    axs[1].plot(train_step, sklearn_results[metric]['train'], color='green', label='training data')
    axs[1].plot(train_step, sklearn_results[metric]['test'], color='red', label='testing data')
    axs[1].set_title('Scikit-learn Random Forest')
    axs[1].set_xlabel('Number of Training Data')
    axs[1].set_ylabel(f'Percent {metric.capitalize()}')
    axs[1].legend()

    #subplot for mlp implementation
    axs[2].plot(epoch_step, mlp_results[metric]['train'], color='green', label='training data')
    axs[2].plot(epoch_step, mlp_results[metric]['test'], color='red', label='testing data')
    axs[2].set_title('Keras Sequential MLP')
    axs[2].set_xlabel('Number of Epochs')
    axs[2].set_ylabel(f'Percent {metric.capitalize()}')
    axs[2].legend()

    #display the figure with the subplots
    plt.tight_layout()
    plt.show()


