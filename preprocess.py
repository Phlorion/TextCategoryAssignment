from imdbDataSet import *
import numpy as np
import pandas as pd

if __name__ == '__main__':
    imdb = IMDB()
    imdb.getTrainingData(1, 2, 3, 20, 50, 200)

    # get feature vector
    features = imdb.getFeatureVector(20, 50)
    features.append("Positive")

    # get values of each feature for n movie reviews
    n = 10
    data = np.zeros((n, len(features)))
    # for the first n reviews
    for i in range(n):
        x_i = imdb.getXtrain(i)
        y_i = imdb.getYtrain(i)
        data[i][len(features)-1] = y_i
        # for word index in x_train
        for wi in x_i:
            if wi == 2:
                continue
            elif (wi in features):
                j = features.index(wi)
                data[i][j] = 1

    
    # create pandas dataframe
    df = pd.DataFrame(data, columns=features)
    # Get the first sample weights
    sample_w = [1/len(df) for i in range(len(df))]
    # add first sample weights to dataframe
    df["Sample weights"] = sample_w
    print(df)