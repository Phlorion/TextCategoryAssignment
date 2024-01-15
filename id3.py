import numpy as np
import math
from statistics import mode

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category

class ID3:
    def __init__(self, features, min_samples_split=2, max_depth=np.inf):
        self.tree = None
        self.features = features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, x, y):
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common, depth=0)
        return self.tree

    def create_tree(self, x_train, y_train, features, category, depth):
        if len(x_train) < self.min_samples_split or len(features) == 0 or depth == self.max_depth:
            return Node(checking_feature=None, is_leaf=True, category=category)

        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)

        igs = [self.calculate_ig(y_train.flatten(), x_train[:, feat_index]) for feat_index in features.flatten()]
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())

        root = Node(checking_feature=max_ig_idx)
        new_features_indices = np.delete(features.flatten(), max_ig_idx)

        for value in [0, 1]:
            x_train_subset = x_train[x_train[:, max_ig_idx] == value]
            y_train_subset = y_train[x_train[:, max_ig_idx] == value].flatten()

            child = self.create_tree(x_train_subset, y_train_subset, new_features_indices, category=m, depth=depth+1)
            if value == 0:
                root.right_child = child
            else:
                root.left_child = child

        return root

    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)
        HC = sum([-list(classes_vector).count(c) / len(classes_vector) * math.log(list(classes_vector).count(c) / len(classes_vector), 2) for c in classes])

        feature_values = set(feature)
        HC_feature = sum([list(feature).count(value) / len(feature) * sum([-classes_of_feat.count(c) / len(classes_of_feat) * math.log(classes_of_feat.count(c) / len(classes_of_feat), 2) for c in classes]) for value in feature_values for classes_of_feat in [[classes_vector[i] for i in range(len(feature)) if feature[i] == value]]])

        return HC - HC_feature

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
