import keras
import imdbDataSet
import numpy as np

imdb = imdbDataSet.IMDB()
imdb.getTrainingData()
x = imdb.x_train
y = imdb.y_train

n1=len(x)
trainx = [x[i] for i in range(0, n1)]   #for which portion of training data do we calculate each possibility
trainy = [y[i] for i in range(0, n1)]   # the results for this portion of data
xtest = imdb.getTrainingData()[1][0]
ytest = imdb.getTrainingData()[1][1]
num_words = 1000        #words in the feature vector
skip = 500              #words we skip

def possibilities():
    n = len(trainx)
    pos = trainy.count(1)/n     #possibility of a positive review
    neg = trainy.count(0)/n     #possibility of a negative review
    return pos,neg

def preprocess(X):        # returns the feature vector of form <0101...0> for items in X
    vec_x = [[1 if w in item else 0 for w in imdb.getFeatureVector(skip,num_words)]for item in X]
    return vec_x

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
        pex[i] = np.log((possumex[i]+1) / (pn+1))   #use the laplace estimator
        pnex[i] = np.log((possumnex[i]+1) / (pn+1))
        nex[i] = np.log((negsumex[i]+1) / (nn+1))
        nnex[i] = np.log((negsumnex[i]+1) / (nn+1))
    # we used the np.log to avoid underflow in the total calculation later
    return pex,pnex,nex,nnex

n2 = len(xtest)
reviews = [xtest[i] for i in range(0,n2)] 
results = [ytest[i] for i in range(0,n2)]

def predict():
    data = preprocess(xtest)
    z = train()
    p1 = z[0]       #P(Xi=1|C=1)
    p2 = z[1]       #P(Xi=0|C=1)
    n1 = z[2]       #P(Xi=1|C=0)
    n2 = z[3]       #P(Xi=0|C=0)
    pos = np.zeros(len(reviews))        #the possibility for each review to be positive
    neg = np.zeros(len(reviews))        #the possibility for each review to be negative
    for i in range(0,len(reviews)):
        pproduct = np.log(possibilities()[0])
        nproduct = np.log(possibilities()[1])
        for j in range(0,len(data[i])):
                if data[i][j] == 1:
                    pproduct += p1[j]
                    nproduct += n1[j]
                elif data[i][j] == 0:
                    pproduct += p2[j]
                    nproduct += n2[j]
        # we add instead of multiplying because we have used logarithms
        pos[i] = pproduct
        neg[i] = nproduct
    
    res = np.zeros(len(reviews))
    for i in range(0,len(reviews)):
        if pos[i] > neg[i]:
            res[i] = 1
    return res

data = predict()
c = 0
for i in range(0,len(data)):
    if data[i] == results[i]:
        c += 1

percentage = (c*100) / len(reviews)
print(percentage)