import keras
import imdbDataSet
import numpy as np
import matplotlib.pyplot as plt

imdb = imdbDataSet.IMDB()
imdb.getTrainingData()
x = imdb.x_train 
y = imdb.y_train
xtest = imdb.getTrainingData()[1][0]
ytest = imdb.getTrainingData()[1][1]

def Bayes(ntrain,X,Y,words=1000,skips=200):
    reviews = X     #reviews used for testing
    results = Y     #expected results of test
    n1=ntrain
    trainx = [x[i] for i in range(0, n1)]   #for which portion of training data do we calculate each possibility
    trainy = [y[i] for i in range(0, n1)]   # the results for this portion of data
    num_words = words        #words in the feature vector
    skip = skips           #words we skip

    def possibilities():
        n = len(trainx)
        pos = trainy.count(1)/n     #possibility of a positive review
        neg = trainy.count(0)/n     #possibility of a negative review
        return pos,neg

    def preprocess(X):        # returns the feature vector of form <0101...0> for items in X
        vec_x = [[1 if w in item else 0 for w in imdb.getFeatureVector(skip,num_words)]for item in X]
        return vec_x

    #train function calculates propability based on training examples
    def train():
        data = preprocess(trainx)           #we need the feature vector of each review
        pn = trainy.count(1)                #count the number of positive reviews
        nn = trainy.count(0)                #count the number of negative reviews

        possumex = np.zeros(num_words)      #for P(Xi=1|C=1) the feature exists in the vector and the category is positive
        possumnex = np.zeros(num_words)     #for P(Xi=0|C=1) the feature does not exist in the vector and the category is positive
        negsumex = np.zeros(num_words)      #for P(Xi=1|C=0) the feature exists in the vector and the category is negative
        negsumnex = np.zeros(num_words)     #for P(Xi=0|C=0) the feature does not exists in the vector and the category is negative

        for col in range(0, num_words):
            for row in range(0,len(trainx)):
                if (data[row][col] == 1) and (trainy[row] == 1):
                    possumex[col] +=1
                elif (data[row][col] == 0) and (trainy[row] == 1):
                    possumnex[col] +=1
                elif (data[row][col] == 1) and (trainy[row] == 0):
                    negsumex[col] += 1
                elif (data[row][col] == 0) and (trainy[row] == 0):
                    negsumnex[col] += 1

        pex = np.zeros(num_words)       #calculate P(Xi=1|C=1)
        pnex = np.zeros(num_words)       #calculate P(Xi=0|C=1)
        nex = np.zeros(num_words)       #calculate P(Xi=1|C=0)
        nnex = np.zeros(num_words)       #calculate P(Xi=0|C=0)

        for i in range(0,num_words):
            pex[i] = np.log((possumex[i]+1) / (pn+1))   #use the Laplace estimator (we add 1 in the category and the total) to avoid 0 propability
            pnex[i] = np.log((possumnex[i]+1) / (pn+1))
            nex[i] = np.log((negsumex[i]+1) / (nn+1))
            nnex[i] = np.log((negsumnex[i]+1) / (nn+1))
        # we used the np.log to avoid underflow in the total propability calculation later
        return pex,pnex,nex,nnex
     
    #predict function takes a table X of reviews and predicts their category
    def predict(rev):
        data = preprocess(rev)            #we create the feature vector of each review in the test data
        z = train()
        p1 = z[0]       #P(Xi=1|C=1)
        p2 = z[1]       #P(Xi=0|C=1)
        n1 = z[2]       #P(Xi=1|C=0)
        n2 = z[3]       #P(Xi=0|C=0)
        pos = np.zeros(len(rev))        #the possibility for each review to be positive
        neg = np.zeros(len(rev))        #the possibility for each review to be negative
        for i in range(0,len(rev)):
            pproduct = np.log(possibilities()[0])       #propability of positive review
            nproduct = np.log(possibilities()[1])       #propability of negative review
            for j in range(0,len(data[i])):
                    if data[i][j] == 1:                 #if the feature exists in the review
                        pproduct += p1[j]               
                        nproduct += n1[j]               #we add P(Xi=1|C=1) and P(Xi=1|C=0) in the total positive and negative propability of the review respectively
                    elif data[i][j] == 0:               #if the feature does not exist in the review
                        pproduct += p2[j]
                        nproduct += n2[j]               #we add P(Xi=0|C=1) and P(Xi=0|C=0) in the total positive and negative propability of the review respectively
            # we add instead of multiplying because we have used logarithms
            pos[i] = pproduct
            neg[i] = nproduct
        
        res = np.zeros(len(rev))            #here we store the predicted result for each review
        for i in range(0,len(rev)):         
            if pos[i] > neg[i]:                 #we compare propabilities
                res[i] = 1
        return res

    
    data = predict(reviews)
    c = 0                                       #total correct predictions for percentage calculation
    for i in range(0,len(data)):
        if data[i] == results[i]:
            c += 1

    percentage = (c*100) / len(reviews)
    return percentage

def Testing(X,Y):           #parameters are either the data we examine (can be train or test)
    res = []
    for i in range(1,11):
        res.append(Bayes(i*1000,X,Y))
    return res

#collect data for learning curve
testx1 = [x[i] for i in range(0, 3000)]  #we first test the accuracy on training data
testy1 = [y[i] for i in range(0, 3000)]

accuracy_train = Testing(testx1,testy1)

testx2 = [xtest[i] for i in range(0, 3000)]  #we now test the accuracy for the test data
testy2 = [ytest[i] for i in range(0, 3000)]

accuracy_test = Testing(testx2,testy2)
axisx = [i*1000 for i in range(1,11)]

plt.xlabel("Review samples")
plt.ylabel("Accuracy")

plt.plot(axisx, accuracy_train, 'green', label="Train Error")
plt.plot(axisx, accuracy_test, 'blue', label="Evaluation Error")

plt.legend()
plt.show()
