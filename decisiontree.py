import numpy as np

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, prediction):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.prediction = prediction
        self.feature_column = 0
        self.feature_value = 0
        self.left = None
        self.right = None


class MyDecisionTree:
    def __init__(self, X, y, max_depth = 100):
        self.X = X
        self.y = y
        self.num_classes = len(set(self.y))
        self.num_features = X.shape[1]
        self.num_samples = X.shape[0]
        self.max_depth = max_depth
        self.tree = None

    def calculate_entropy(self):
        #still have to fix - entropy isn't incorporated in training code yet

        # counts is an array of how many of each label
        _, counts = np.unique(self.y, return_counts=True)

        # gets probabilities
        probabilities = counts / counts.sum()

        # calculates entropy
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy

    def gini(self, num, den):
        #gini = 1 - sum(probabilities^2)
        probabilities = []
        for x in range(self.num_classes):
            probabilities.append((num[x]/den) ** 2)

        gini = 1.0 - sum(probabilities)
        return gini

    def best_split(self, X, y):

        samples_left = len(y)
        if samples_left <= 1:
            return None, None
        _, counts = np.unique(self.y, return_counts=True)
        counts = list(counts)
        best_gini = self.gini(counts, samples_left)
        #print("best gini ", best_gini)

        best_feature, best_value = None, None

        #iterates through features
        for feature in range(self.num_features):
            #list1 = self.X[:, feature] array of values at feature column
            #list2 = self.y
            #feature_column_values is tuple of list1 sorted
            #labels is tuple of list2 sorted
            feature_column_values, labels = zip(*sorted(zip(X[:, feature], y.astype(np.int64))))

            #creates an array of length num_classes filled with zeros
            num_left = [0] * self.num_classes


            #array containing amount of each sample [class1 class2]
            num_right = counts.copy()

            #for each feature iterates through each sample
            for i in range (1, samples_left):
                if feature_column_values[i] == feature_column_values[i - 1]:
                    continue
                #c starts at last element in array
                #c is y[i-1]
                #c is label at given sample
                #c can range from 1 to total samples
                c = labels[i-1]

                #left starts at index [0]
                #right starts at index [-2] - second to last element

                #[] of length of classes
                num_left[c] += 1

                #[] of length of classses
                num_right[c] -=1

                gini_left = self.gini(num_left, i)

                gini_right = self.gini(num_right, (samples_left-i))

                #weighted average
                gini = (i * gini_left + (samples_left-i) * gini_right) / samples_left

                #if new gini is < best gini update
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_value = (feature_column_values[i] + feature_column_values[i - 1]) / 2  # midpoint

        return best_feature, best_value



    def predict(self, X):
        prediction = [self._predict(inputs) for inputs in X]
        #print(prediction)
        return prediction

    def _predict(self, inputs):
        #predict class for a single sample
        #subtree
        node = self.tree
        while node.left:
            if inputs[node.feature_column] < node.feature_value:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def grow_tree(self, X=None, y=None, depth=0):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        #_, counts = np.unique(y, return_counts=True)
        #counts = list(counts)
        counts = [np.sum(y == i) for i in range(self.num_classes)]
        #print("counts: ", counts)
        prediction = np.argmax(counts)
        if prediction == 0:
            prediction = -1
        node = Node(gini = self.gini(counts, y.size), num_samples=y.size,
                    num_samples_per_class= counts, prediction= prediction)
        if depth < self.max_depth:
            feature, value = self.best_split(X, y)

            if feature is not None:
                #print("double inside")
                #all samples in X where value for feature feature is < value
                left_feature = X[:, feature] < value
                #lefts hold everything in left
                X_left, y_left = X[left_feature], y[left_feature]
                #right holds complement
                X_right, y_right = X[~left_feature], y[~left_feature]

                node.feature_column = feature
                node.feature_value = value
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)

        return node


    def fit(self):
        #self.tree returns node
        self.tree = self.grow_tree(depth=10)
        return self.tree

