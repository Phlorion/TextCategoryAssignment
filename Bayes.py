import keras
import imdbDataSet
import numpy as np

imdb = imdbDataSet.IMDB()
imdb.getTrainingData()

x = imdb.x_train
y = imdb.y_train

#calculate the possibility of C=c for each category 
def possibilities():
    sump = 0
    for i in y:
        if i == 1:
            sump += 1
    n = len(x)
    posp = sump/n           #possibility that the review is possitive
    posn = 1 - posp         #possibility that the review is negative
    return posp, posn


positive = possibilities()[0]
negative = possibilities()[1]


def TrainVector(n,num_of_words):
    features = imdb.getFeatureVector(100, num_of_words)
    data = np.zeros((n, len(features)))
    # for the first n reviews
    for i in range(n):
        x_i = imdb.getXtrain(i)
        y_i = imdb.getYtrain(i)
        #data[i][len(features)-1] = y_i
        #for word index in x_train
        for wi in x_i:
            if wi == 2:
                continue
            elif (wi in features):
                j = features.index(wi)
                data[i][j] = 1
    return data


def train():
    cat = set(y)        #we create a set of possible categories (here there is 2 with identifiers 1 and 0)
    z = set([i for j in range(0,len(x)) for i in x[j] ])    #we create a set of values in every review for X=x calculation
    num_words = 1000
    data = TrainVector(len(x), num_words)
    possumex = np.zeros(num_words)      #for P(X=1|C=1)
    possumnex = np.zeros(num_words)     #for P(X=0|C=1)
    negsumex = np.zeros(num_words)      #for P(X=1|C=0)
    negsumnex = np.zeros(num_words)     #for P(X=0|C=0)
    
    for col in range(0, num_words):
        for row in range(0,len(x)):
            if (data[row][col] == 1) and (y[row] == 1):
                possumex[col] +=1
            elif (data[row][col] == 0) and (y[row] == 1):
                possumnex[col] +=1
            elif (data[row][col] == 1) and (y[row] == 0):
                negsumex[col] += 1
            elif (data[row][col] == 0) and (y[row] == 0):
                negsumnex[col] += 1
    
    pn = list(y).count(1)
    nn = list(y).count(0)
    
    pex = np.zeros(num_words)       #calculate P(X=1|C=1)
    for i in range(0,num_words):
       pex[i] = possumex[i] / pn    

    pnex = np.zeros(num_words)       #calculate P(X=0|C=1)
    for i in range(0,num_words):
       pnex[i] = possumex[i] / pn

    nex = np.zeros(num_words)       #calculate P(X=1|C=0)
    for i in range(0,num_words):
       nex[i] = negsumex[i] / nn

    nnex = np.zeros(num_words)       #calculate P(X=0|C=0)
    for i in range(0,num_words):
       nnex[i] = negsumex[i] / nn
    
    return pex,pnex,nex,nnex
    
xtest = imdb.getTrainingData()[1][0]
ytest = imdb.getTrainingData()[1][1]

def Vector(n,num_of_words,X):
    features = imdb.getFeatureVector(100, num_of_words)
    data = np.zeros((n, len(features)))
    for i in range(n):
        x_i = X[i]
        #data[i][len(features)-1] = y_i
        #for word index in x_train
        for wi in x_i:
            if wi == 2:
                continue
            elif (wi in features):
                j = features.index(wi)
                data[i][j] = 1
    return data


def predict():
    num_words = 1000
    data = Vector(len(xtest),num_words,xtest)
    pos = np.zeros(len(xtest))
    neg = np.zeros(len(xtest))
    possible = train()

    for i in range(0,len(xtest)):
       pproduct = positive
       nproduct = negative
       for j in range(0,len(data[i])):
            if data[i][j] == 1:
               pproduct *= possible[0][j]
               nproduct *= possible[2][j]
            elif data[i][j] == 0:
               pproduct *= possible[1][j]
               nproduct *= possible[3][j]
       pos[i] = pproduct
       neg[i] = nproduct
    return pos,neg


def test():
    z = predict()
    z1 = z[0]
    z2 = z[1]
    category = 0
    for i in range(0,len(x)):
        if z1[i] > z2[i]:
            category = 1
        print(category == ytest[i])  

test()       