import keras

class IMDB:
    def __init__(self):
        self.word_index = None
        self.inverted_word_index = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    # Use the default parameters to keras.datasets.imdb.load_data
    def getTrainingData(self, start_char=1, oov_char=2, index_from=3, skip_top=100, num_words=1000, maxlen=200):
        # Retrieve the training sequences.
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
            start_char=start_char, oov_char=oov_char, index_from=index_from, skip_top=skip_top, num_words=num_words, maxlen=maxlen
        )

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        
        # Retrieve the word index file mapping words to indices
        self.word_index = keras.datasets.imdb.get_word_index()
        # Reverse the word index to obtain a dict mapping indices to words
        # And add `index_from` to indices to sync with `x_train`
        self.inverted_word_index = dict(
            (i + index_from, word) for (word, i) in self.word_index.items()
        )
        # Update `inverted_word_index` to include `start_char` and `oov_char`
        self.inverted_word_index[start_char] = "[START]"
        self.inverted_word_index[oov_char] = "[OOV]"

        return (x_train, y_train), (x_test, y_test)

    def getFeatureVector(self, skip_top=100, num_words=1000):
        if (self.word_index == None):
            print("Empty dataset")
            return
        
        feature_vector = [k for k in range(skip_top+1, skip_top+num_words+1, 1)]
        return feature_vector

    # Decode the first sequence in the dataset
    def getDecodedSequence(self, x, n):
        decoded_sequence = " ".join(self.inverted_word_index[i] for i in x[n])
        return decoded_sequence
    
    def getXtrain(self, n):
        return self.x_train[n]
    
    def getYtrain(self, n):
        return self.y_train[n]
    
    def getXtest(self, n):
        return self.x_test[n]
    
    def getYtest(self, n):
        return self.y_test[n]
    
    def getWordIndex(self, word):
        return self.word_index[word]
    
    def getInvertedWordIndex(self, n):
        return self.inverted_word_index[n+4]
    

if __name__ == '__main__':
    imdb = IMDB()
    (x_train, y_train), _ = imdb.getTrainingData()
    print(imdb.getDecodedSequence(x_train, 0))
    print(imdb.getXtrain(0))
    print(imdb.getYtrain(0))

    print(imdb.getInvertedWordIndex(0))


    