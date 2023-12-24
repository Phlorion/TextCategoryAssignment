from statistics import mode
import numpy as np
import math

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
        


class ID3:
    def __init__(self, features):
        self.tree = None
        self.features = features
    
    def fit(self, x, y):
        '''
        creates the tree
        '''
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common)
        return self.tree
    
    def create_tree(self, x_train, y_train, features, category):
        
        # check empty data
        if len(x_train) == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
        
        # check all examples belonging in one category
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
        
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        
        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))
        
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, 
                                           category=m)  # go left for X = 1
        
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m)  # go right for X = 0
        
        return root


    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    

        

    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x:  # for every example 
            tmp = self.tree  # begin at root
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)


            