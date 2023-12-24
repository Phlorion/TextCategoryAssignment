from id3  import *
from imdbDataSet import *
import numpy as np
import keras as kr

#Ορισμός υπερπαραμέτρων
tree_number = 10
fv_skip_top = 200
fv_num_words = 5000 

#obtain imdb data 
imdb = IMDB()
imdb.getTrainingData(skip_top=fv_skip_top, num_words=fv_num_words)

#create encoded feature vector (index of words)
feature_vector = imdb.getFeatureVector(skip_top=fv_skip_top, num_words=fv_num_words)

