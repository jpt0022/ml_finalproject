import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split

def load_data(datafile):
    """
    @params - takes in a datafile
    @return - returns a X, y tuple with
                X containing features and y containing lables

    reads in datafile and parses it separating the features and the label
    """
    #reads data into pandas dataframe
    df = pd.read_csv(datafile)

    #converts into list
    data = df.values.tolist()

    #Splits array into features and labels
    #X grabs everything except the last index - all the features
    #y grabs the last index - all the labels
    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    y[y < 1] = -1
    return X, y

def split_data(X, y):
    """
        @params - takes in an X, y tuple
        @return - returns an X_train, X_test, y_train, y_test set

        splits the training set 75/25 into training and testing set
        random_state = 24 set to generate the same test set each time for consistency
        """

    #split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=24)
    return X_train, X_test, y_train, y_test
"""
def main():
    X,y = load_data("spambase.data")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)

if __name__ == "__main__":
    main()
"""