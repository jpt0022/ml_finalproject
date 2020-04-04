import numpy as np

def get_potentional_splits(X):

    potential_splits = {}
    _, n_attributes = X.shape

    for attribute in range(n_attributes):
        potential_splits[attribute] = []
        values = X[:, attribute]
        #returns a sorted array of unique elements
        unique_values = np.unique(values)
        for idx in range(len(unique_values)):
           if idx != 0:
               #skips first element
               current_value = unique_values[idx]
               #skips last element
               previous_value = unique_values[idx-1]
               potential_split = (current_value + previous_value) / 2

               potential_splits[attribute].append(potential_split)
    return potential_splits

def split_data(X, y, attribute, attribute_value):
    """
    returns data to the left(below) and data to the right(above)
    """

    y_left = []
    y_right = []
    #returns an array that contains values of specified attribute column
    split_attribute_values = X[:, attribute]
    X_left = X[split_attribute_values <= attribute_value]
    X_right = X[split_attribute_values > attribute_value]
    for idx in range(len(split_attribute_values)):
        if split_attribute_values[idx] <= attribute_value:
            y_left.append(y[idx])
        else:
            y_right.append(y[idx])

    #X_left = X[split_attribute_values <= attribute_value]
    #X_right = X[split_attribute_values > attribute_value]

    return X_left, X_right, y_left, y_right

def calculate_entropy(y):
    #counts is an array of how many of each label
    _, counts = np.unique(y, return_counts=True)

    #gets probabilities
    probabilities = counts / counts.sum()

    #calculates entropy
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def calculate_overall_entropy(X_left, X_right, y_left, y_right):
    num_samples = len(X_left) + len(X_right)
    p_left = len(X_left)/num_samples
    p_right = len(X_right)/num_samples
    overall_entropy = (p_left * calculate_entropy(y_left) * p_right * calculate_entropy(y_right))
    return overall_entropy

def determine_best_split(X, y, potential_splits):
    overall_entropy = 99999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            X_left, X_right, y_left, y_right = split_data(X, y, column_index, value)
            current_overall_entropy = calculate_overall_entropy(X_left, X_right, y_left, y_right)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value

def decision_tree_algorithm(X, y):
    print(len(X))
    print(len(y))
    if len(X) == 1:
        return y

    potential_splits = get_potentional_splits(X)
    split_column, split_value = determine_best_split(X, y, potential_splits)
    X_left, X_right, y_left, y_right = split_data(X, y, split_column, split_value)

    question = "{} <= {}".format(split_column, split_value)
    sub_tree = {question: []}

    yes_answer = decision_tree_algorithm(X_left,y_left)
    no_answer = decision_tree_algorithm(X_left, y_left)

    sub_tree[question].append(yes_answer)
    sub_tree[question].append(no_answer)
    return sub_tree

