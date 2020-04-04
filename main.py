import sklearn as sk
import read_data
import numpy as np
import pandas
#import dttry
import decisiontree
import svm
import sklearn.tree as tree
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn import tree
from decisiontree import MyDecisionTree
from sklearn.metrics import accuracy_score


X, y = read_data.load_data("spambase.data")
X_train, X_test, y_train, y_test = read_data.split_data(X, y)


#print("X_train shape", X_train.shape)
#print("y_train shape", y_train.shape)
#print("X_test shape", X_test.shape)
#print("y_test shape", y_test.shape)

#print(y_test)
my_dt = MyDecisionTree(X_train, y_train)
dt = my_dt.fit()
my_dt_predict = my_dt.predict(X_test)
print("MyDecisionTree accuracy: ", accuracy_score(y_test, my_dt_predict)*100)

#clf = tree.DecisionTreeClassifier()
#clf.fit(X_train, y_train)
#sk_y_predict = clf.predict(X_test)
#print("Sklearn accuracy: ", accuracy_score(y_test, sk_y_predict)*100)

weights = svm.pegasos(X_train, y_train, lam=100)
my_svm_predict = svm.test_svm(X_test, weights)
print("MySVM accuracy: ", accuracy_score(y_test, my_svm_predict)*100)

weights_k = svm.kernelized_pegasos(X_train, y_train, lam=1000)
my_svm_predict_kernel = svm.test_svm(X_test, weights_k)
print("MySVMKernel accuracy: ", accuracy_score(y_test, my_svm_predict_kernel)*100)


#dot_data = StringIO()
#tree.export_graphviz(dt, out_file=dot_data)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
