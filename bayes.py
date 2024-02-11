import keras
import tensorflow as tf
import imdbDataSet
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

imdb = imdbDataSet.IMDB()
d = imdb.getTrainingData()
x = imdb.x_train 
y = imdb.y_train
xtest = d[1][0]
ytest = d[1][1]

def preprocess(T,skips=200,words=1000):        # returns the feature vector in the form of <0101...0110> for items in X
    vec_x = [[1 if w in item else 0 for w in imdb.getFeatureVector(skips,words)]for item in T]
    return vec_x
#------------------------------------------------------------Part A: My Naive Bayes---------------------------------------------------------------------------------------------------------------
def Bayes(ntrain,X,Y,words=1000,skips=200):
    reviews = X     #reviews used for testing (either train or test data)
    results = Y     #expected results of test
    n1 = ntrain       #number of train data we will use
    trainx = [x[i] for i in range(0, n1)]   #for which portion of train data do we calculate each possibility
    trainy = [y[i] for i in range(0, n1)]   # the results for this portion of data
    num_words = words        #words in the feature vector
    skip = skips             #words we skip

    def possibilities():
        n = len(trainx)
        pos = sum(trainy)/n     #possibility of a positive review (the number of positive reviews is the sum of the elements in the array since they have the value of 1)
        neg = 1-pos     #possibility of a negative review (1's complementary)
        return pos,neg

    prob = possibilities()
    positive = np.log(prob[0])      #we take the logarithm of these two propabilities
    negative = np.log(prob[1])
    train_data = preprocess(trainx,skip,num_words)           #we need the feature vector of each review  
    
    #train function calculates propabilities based on training examples
    def train():     
        pn = sum(trainy)                  #count the number of positive reviews (same way as before)
        nn = len(trainy)-pn               #count the number of negative reviews (total reviews - positive ones)
        possumex = np.zeros(num_words)      #for P(Xi=1|C=1) the feature exists in the vector and the category is positive
        possumnex = np.zeros(num_words)     #for P(Xi=0|C=1) the feature does not exist in the vector and the category is positive
        negsumex = np.zeros(num_words)      #for P(Xi=1|C=0) the feature exists in the vector and the category is negative
        negsumnex = np.zeros(num_words)     #for P(Xi=0|C=0) the feature does not exists in the vector and the category is negative

        for col in range(0, num_words):                             #for each feature we count the number of positive reviews it has appeared (or not). we do the same for the negative ones
            for row in range(0,len(train_data)):
                if (train_data[row][col] == 1) and (trainy[row] == 1):         #positive review inludes the feature
                    possumex[col] +=1
                elif (train_data[row][col] == 0) and (trainy[row] == 1):        #positive review does not include the feature
                    possumnex[col] +=1
                elif (train_data[row][col] == 1) and (trainy[row] == 0):        #negative review inludes the feature
                    negsumex[col] += 1
                elif (train_data[row][col] == 0) and (trainy[row] == 0):        #negative review does not include the feature
                    negsumnex[col] += 1

        pex = np.zeros(num_words)       #calculate P(Xi=1|C=1) for each feature in the vector
        pnex = np.zeros(num_words)       #calculate P(Xi=0|C=1)
        nex = np.zeros(num_words)       #calculate P(Xi=1|C=0)
        nnex = np.zeros(num_words)       #calculate P(Xi=0|C=0)

        for i in range(0,num_words):
            pex[i] = np.log((possumex[i]+1) / (pn+2))       #we use the laplace smoothing. we add 2 more examples for each category, 1 where the feature is included and 1 where it is not
            pnex[i] = np.log((possumnex[i]+1) / (pn+2))
            nex[i] = np.log((negsumex[i]+1) / (nn+2))
            nnex[i] = np.log((negsumnex[i]+1) / (nn+2))
        # we used the np.log to avoid underflow in the total propability calculation later
        return pex,pnex,nex,nnex
    #predict function takes a table X of reviews and predicts their category
    def predict(rev):
        data = preprocess(rev,skip,num_words)            #we create the feature vector of each review in the test data
        z = train()     #we call the train function to get the propabilities
        p1 = z[0]       #P(Xi=1|C=1)
        p2 = z[1]       #P(Xi=0|C=1)
        n1 = z[2]       #P(Xi=1|C=0)
        n2 = z[3]       #P(Xi=0|C=0)
        res = np.zeros(len(rev))            #here we store the predicted result for each review (initially all zeros)
        for i in range(0,len(rev)):
            pproduct = positive      #propability of positive review
            nproduct = negative       #propability of negative review
            for j in range(0,len(data[i])):
                    if data[i][j] == 1:                 #if the feature exists in the review
                        pproduct += p1[j]               
                        nproduct += n1[j]               #we add P(Xi=1|C=1) and P(Xi=1|C=0) in the total positive and negative propability of the review respectively
                    elif data[i][j] == 0:               #if the feature does not exist in the review
                        pproduct += p2[j]
                        nproduct += n2[j]               #we add P(Xi=0|C=1) and P(Xi=0|C=0) in the total positive and negative propability of the review respectively
            # we add instead of multiplying because we have used logarithms    
            if pproduct> nproduct:                 #we compare propabilities
                res[i] = 1                          #if the propability of the review belonging in the positive category is greater than the negative 1, we change 0 to 1 in the array of results
        return res          #return the predictions

    data = predict(reviews)             #get the predictions for the test data and calculate:
    percentage = sklearn.metrics.accuracy_score(y_true=results, y_pred=data)    #accuracy
    precision = sklearn.metrics.precision_score(y_true=results, y_pred=data)    #precision
    recall = sklearn.metrics.recall_score(y_true=results, y_pred=data)          #recall
    f1 = sklearn.metrics.f1_score(y_true=results, y_pred=data)                  #F1 score
    return percentage, precision, recall, f1

#test my Naive Bayes
accuracy_train = []
precision_train = []
recall_train = []
f1_train = []
accuracy_test = []
precision_test = []
recall_test = []
f1_test = []

testx2 = [xtest[i] for i in range(0,3000)]          #when we evaluate test data we keep their number constant
testy2 = [ytest[i] for i in range(0,3000)]
for i in range(1,15):
    #for train data
    testx1 = [x[i] for i in range(0,i*1000)]        #evaluation of train data (we evaluate the exact same data we used for training)
    testy1 = [y[i] for i in range(0,i*1000)]        #their expected answers
    results_train = Bayes(i*1000,testx1,testy1)     #call the Bayes function for train data
    accuracy_train.append(results_train[0])
    precision_train.append(results_train[1])
    recall_train.append(results_train[2])
    f1_train.append(results_train[3])

    #for test data
    results_test = Bayes(i*1000,testx2,testy2)      #call the Bayes function for test data
    accuracy_test.append(results_test[0])
    precision_test.append(results_test[1])
    recall_test.append(results_test[2])
    f1_test.append(results_test[3])
#'''
axisx = [i*1000 for i in range(1,15)]

#plot accuracy curve
plt.xlabel("Review samples")
plt.ylabel("Accuracy")

plt.plot(axisx, accuracy_train, 'green', label="Train")
plt.plot(axisx, accuracy_test, 'blue', label="Evaluation")

plt.legend()
plt.show()

#plot precision curve
plt.xlabel("Review samples")
plt.ylabel("Precision")

plt.plot(axisx, precision_train, 'green', label="Train Data")
plt.plot(axisx, precision_test, 'blue', label="Test Data")

plt.legend()
plt.show()

#plot recall curve
plt.xlabel("Review samples")
plt.ylabel("Recall")

plt.plot(axisx, recall_train, 'green', label="Train Data")
plt.plot(axisx, recall_test, 'blue', label="Test Data")

plt.legend()
plt.show()

#plot F1 curve
plt.xlabel("Review samples")
plt.ylabel("F1")

plt.plot(axisx, f1_train, 'green', label="Train Data")
plt.plot(axisx, f1_test, 'blue', label="Test Data")

plt.legend()
plt.show()

#Result table
print ("------------------------------Evaluate Train Data--------------------------------")
print("# train data   " + "     Accuracy  " + "                 Precision  " + "                  Recall  " + "                F1 Score  ")
for i in range(0,14):
    print(str((i+1)*1000) + "           " + str("{:.16f}".format(accuracy_train[i])) + "          " + str("{:.16f}".format(precision_train[i])) + "         " + str("{:.16f}".format(recall_train[i])) + "         " + str("{:.16f}".format(f1_train[i])))

print ("\n-----------------------------Evaluate Test Data--------------------------------")
print("# train data   " + "      Accuracy  " + "                Precision  " + "                Recall  " + "                F1 Score  ")
for i in range(0,14):
    print(str((i+1)*1000) + "            " + str("{:.16f}".format(accuracy_test[i])) + "        " + str("{:.16f}".format(precision_test[i])) + "        " + str("{:.16f}".format(recall_test[i])) + "         " + str("{:.16f}".format(f1_test[i])))#'''

#'''
#--------------------------------------------------------------------Part B: Comparison with sklearn Bayes--------------------------------------------------------------------------------------
def Bayes2Compare(ntrain,X,Y,words=1000,skips=200):
    reviews = preprocess(X,skips,words)     #reviews used for testing
    results = Y     #expected results of test
    n1 = ntrain     #number of train data used
    trainx = preprocess([x[i] for i in range(0, n1)],skips,words)  #for which portion of training data do we calculate each possibility
    trainy = [y[i] for i in range(0, n1)]   # the results for this portion of data
    
    nb = GaussianNB()
    nb.fit(trainx, trainy)
    data = nb.predict(reviews)
    percentage = sklearn.metrics.accuracy_score(y_true=results, y_pred=data)
    precision = sklearn.metrics.precision_score(y_true=results, y_pred=data) 
    recall= sklearn.metrics.recall_score(y_true=results, y_pred=data)
    f1 = sklearn.metrics.f1_score(y_true=results, y_pred=data)
    return percentage, precision, recall, f1

#test sklearn Bayes
accuracy_train = []
precision_train = []
recall_train = []
f1_train = []
accuracy_test = []
precision_test = []
recall_test = []
f1_test = []

testx2 = [xtest[i] for i in range(0,3000)]          #again we keep the number of test data constant 
testy2 = [ytest[i] for i in range(0,3000)]
for i in range(1,15):
    #for train data
    testx1 = [x[i] for i in range(0,i*1000)]        #evaluation of train data (we evaluate the exact same data we used for training)
    testy1 = [y[i] for i in range(0,i*1000)]        #their expected answers
    results_train = Bayes2Compare(i*1000,testx1,testy1)     #call the sklearn Naive Bayes for train data
    accuracy_train.append(results_train[0])
    precision_train.append(results_train[1])
    recall_train.append(results_train[2])
    f1_train.append(results_train[3])

    #for test data
    results_test = Bayes2Compare(i*1000,testx2,testy2)      #call the sklearn Naive Bayes for test data
    accuracy_test.append(results_test[0])
    precision_test.append(results_test[1])
    recall_test.append(results_test[2])
    f1_test.append(results_test[3])
#'''
axisx = [i*1000 for i in range(1,15)]

#plot accuracy curve
plt.xlabel("Review samples")
plt.ylabel("Accuracy")

plt.plot(axisx, accuracy_train, 'green', label="Train")
plt.plot(axisx, accuracy_test, 'blue', label="Evaluation")

plt.legend()
plt.show()

#plot precision curve
plt.xlabel("Review samples")
plt.ylabel("Precision")

plt.plot(axisx, precision_train, 'green', label="Train Data")
plt.plot(axisx, precision_test, 'blue', label="Test Data")

plt.legend()
plt.show()

#plot recall curve
plt.xlabel("Review samples")
plt.ylabel("Recall")

plt.plot(axisx, recall_train, 'green', label="Train Data")
plt.plot(axisx, recall_test, 'blue', label="Test Data")

plt.legend()
plt.show()

#plot F1 curve
plt.xlabel("Review samples")
plt.ylabel("F1")

plt.plot(axisx, f1_train, 'green', label="Train Data")
plt.plot(axisx, f1_test, 'blue', label="Test Data")

plt.legend()
plt.show()

#Result table
print ("------------------------------Evaluate Train Data--------------------------------")
print("# train data   " + "     Accuracy  " + "                 Precision  " + "                  Recall  " + "                F1 Score  ")
for i in range(0,14):
    print(str((i+1)*1000) + "           " + str("{:.16f}".format(accuracy_train[i])) + "          " + str("{:.16f}".format(precision_train[i])) + "         " + str("{:.16f}".format(recall_train[i])) + "         " + str("{:.16f}".format(f1_train[i])))

print ("\n-----------------------------Evaluate Test Data--------------------------------")
print("# train data   " + "      Accuracy  " + "                Precision  " + "                Recall  " + "                F1 Score  ")
for i in range(0,14):
    print(str((i+1)*1000) + "            " + str("{:.16f}".format(accuracy_test[i])) + "        " + str("{:.16f}".format(precision_test[i])) + "        " + str("{:.16f}".format(recall_test[i])) + "         " + str("{:.16f}".format(f1_test[i])))
#'''

#'''
#-------------------------------------------------------------Part C: Comparison with MLPs--------------------------------------------------------------------------------------------------------
VOCAB_SIZE = 15000
MAX_SEQUENCE_LENGTH = 250

#create the rnn with gru layers sequential model
imdb_rnn_with_gru = keras.models.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, output_dim = 32, input_length = MAX_SEQUENCE_LENGTH), #embedding layer
    keras.layers.GRU(units=32),#gru layer 1
    keras.layers.Dense(units=1, activation='sigmoid')  #output layer, output = probability of classification in the positive category
])

#compile the model
imdb_rnn_with_gru.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

def runTestMLP(N=250, M=1000, Train=1000, TESTS=1000):
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.getTrainingData(skip_top=N, num_words=M)

    X_train = x_train_raw[:Train]
    X_test = x_test_raw[:TESTS]
    y_train = y_train_raw[:Train]
    y_test = y_test_raw[:TESTS]

    #pad data sequences to fit into a specific size
    rnn_train_data = keras.utils.pad_sequences(sequences=X_train, maxlen=MAX_SEQUENCE_LENGTH)
    rnn_test_data = keras.utils.pad_sequences(sequences=X_test, maxlen=MAX_SEQUENCE_LENGTH)

    #train the rnn model, we use a constant number of epochs to compare with the other implementations, the only variable here is the training data 
    imdb_rnn_with_gru.fit(rnn_train_data, y_train, epochs=10, batch_size=1024, validation_split=0.2)

    #obtain rnn model's predictions, if the probability is above 0.5 we consider it a positive classification, negative otherwise.
    #on training predictions
    y_pred_train_rnn = (imdb_rnn_with_gru.predict(rnn_train_data) > 0.5).astype("int32")
    #on testing predictions
    y_pred_test_rnn = (imdb_rnn_with_gru.predict(rnn_test_data) > 0.5).astype("int32")

    y_pred_train_rnn = y_pred_train_rnn.flatten()
    y_pred_test_rnn = y_pred_test_rnn.flatten()

    return y_train, y_test, y_pred_train_rnn, y_pred_test_rnn
# run Test SK
_xs = []
for i in range(0,14):
    _xs.append(1000*(i+1))

trains_acc = []
tests_acc = []
train_pre = []
test_pre = []
train_rec = []
test_rec = []
train_f1 = []
test_f1 = []

for i in range(0,14):
    y_train, y_test, y_train_pred, y_test_pred = runTestMLP(N=200,M=1000, Train=1000*(i+1),TESTS=3000)

    res_train = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    res_test = sklearn.metrics.accuracy_score(y_test, y_test_pred)
    res_precision_train = sklearn.metrics.precision_score(y_true=y_train, y_pred=y_train_pred)
    res_precision_test = sklearn.metrics.precision_score(y_true=y_test, y_pred=y_test_pred)
    res_recall_train = sklearn.metrics.recall_score(y_true=y_train, y_pred=y_train_pred)
    res_recall_test = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_test_pred)
    res_f1_train = sklearn.metrics.f1_score(y_true=y_train, y_pred=y_train_pred)
    res_f1_test = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_test_pred)

    trains_acc.append(res_train)
    tests_acc.append(res_test)
    train_pre.append(res_precision_train)
    test_pre.append(res_precision_test)
    train_rec.append(res_recall_train)
    test_rec.append(res_recall_test)
    train_f1.append(res_f1_train)
    test_f1.append(res_f1_test)


#Calculate the loss and val-loss
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.getTrainingData(skip_top=200, num_words=1000)
x_train = x_train_raw[:10000]
y_train = y_train_raw[:10000]

#gather loss data from rnn, increasing epochs with constant train and test data for each loop 
#compile the model again
imdb_rnn_with_gru_new = keras.models.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, output_dim = 32, input_length = MAX_SEQUENCE_LENGTH), #embedding layer
    keras.layers.GRU(units=32),#gru layer 1
    keras.layers.Dense(units=1, activation='sigmoid')  #output layer, output = probability of classification in the positive category
])

imdb_rnn_with_gru_new.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

rnn_train_data2 = keras.utils.pad_sequences(sequences=x_train, maxlen=MAX_SEQUENCE_LENGTH)

rnn_history = imdb_rnn_with_gru_new.fit(rnn_train_data2, y_train_raw, epochs=15, batch_size=1024, validation_split=0.2)
print(rnn_history.history['loss'])
print(rnn_history.history['val_loss'])

# plot the accuracy curve
plt.xlabel("Review samples")
plt.ylabel("Accuracy")

plt.plot(_xs, trains_acc, 'green', label="Train")
plt.plot(_xs, tests_acc, 'blue', label="Evaluation")

plt.legend()
plt.show()

# plot the recall curve
plt.xlabel("Review samples")
plt.ylabel("Recall")

plt.plot(_xs, train_rec, color='green', label="Train")
plt.plot(_xs, test_rec, color='blue', label="Test")

plt.legend()
plt.show()

# plot the precision curve
plt.xlabel("Review samples")
plt.ylabel("Precision")

plt.plot(_xs, train_pre, color='green', label="Train")
plt.plot(_xs, test_pre, color='blue', label="Test")

plt.legend()
plt.show()

# plot the f1 score
plt.title("F1 Score")
plt.xlabel("Review samples")
plt.ylabel("F1")

plt.plot(_xs, train_f1, 'green', label="Train")
plt.plot(_xs, test_f1, 'blue', label="Test")

plt.legend()
plt.show()

# rnn loss
plt.xlabel("Epochs")
plt.ylabel("Loss")

epochs = np.linspace(1, 15, 15)
plt.plot(epochs, rnn_history.history['loss'], 'blue', label="loss")
plt.plot(epochs, rnn_history.history['val_loss'], 'green', label="val-loss")

plt.legend()
plt.show()

#Table of results
print ("------------------------------Evaluate Train Data--------------------------------")
print("# train data   " + "     Accuracy  " + "                 Precision  " + "                  Recall  " + "                F1 Score  ")
for i in range(0,14):
    print(str((i+1)*1000) + "           " + str("{:.16f}".format(trains_acc[i])) + "          " + str("{:.16f}".format(train_pre[i])) + "         " + str("{:.16f}".format(train_rec[i])) + "         " + str("{:.16f}".format(train_f1[i])))

print ("\n-----------------------------Evaluate Test Data--------------------------------")
print("# train data   " + "      Accuracy  " + "                Precision  " + "                Recall  " + "                F1 Score  ")
for i in range(0,14):
    print(str((i+1)*1000) + "            " + str("{:.16f}".format(tests_acc[i])) + "        " + str("{:.16f}".format(test_pre[i])) + "        " + str("{:.16f}".format(test_rec[i])) + "         " + str("{:.16f}".format(test_f1[i])))

loss = rnn_history.history['loss']
val_loss = rnn_history.history['val_loss']

print("\n---------------------Loss & Val-loss--------------------------------------")
print("# epochs  " + "  Loss  " + "  Val-loss  ")
for i in range(len(loss)):
    print(str(i+1) + "          " + str("{:.4f}".format(loss[i])) + "    " + str("{:.4f}".format(val_loss[i])))
#'''