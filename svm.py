import numpy as np
import random
import math
from numpy.linalg import norm




"""
def compute_cost(X, y, weights, lam):

    #m = number of samples
    m= X.shape[0]

    distances = 1 - (y * (np.dot(X, weights))

    #take max between zero and l(w;(x,y))
    distances[distances < 0] = 0

    #summation divided by number of samples
    hinge_loss = (np.sum(distances)/m)

    cost = lam/2 * np.dot(weights, weights) + hinge_loss
    return cost
"""

#add kernalized pegasos




def cost_gradient(X_batch, y_batch, weights, lam, ):

    #distance = 1 - (y_batch * (np.dot(X_batch, weights) + bias))
    #dw = np.zeros(len(weights))
    indicator = 0
    L = y_batch * np.dot(X_batch, weights)

    if L < 1:
        indicator = 1

    dw = lam * weights - indicator * np.dot(y_batch, X_batch)
    return dw

def projection_step(weights, lam):
    ps = (1/np.sqrt(lam)) / norm(weights)
    minimize = min(1, ps)
    return minimize*weights
np
def kernel(X, y, variance=1):
    l2 = norm(X-y)**2
    #rbf kernel
    kernel = np.exp(-l2 / (2*(variance **2)) )
    return kernel


def pegasos(X, y, lam):
    T = 1000
    #array of zeros with length = num_features
    weights = np.zeros(X.shape[1])
    # stochastic gradient descent
    for t in range(1, T):
        #on iteration t choose a random sample (x, y) from index (1 to m)
        rand_idx = random.randrange(0, X.shape[0])
        nth = 1/(lam*t)
        #calculate obj func
        #obj_func = compute_cost(x[rand_idx], y[rand_idx], weights, lam)
        subgradient = cost_gradient(X[rand_idx], y[rand_idx], weights, lam)

        weights += weights + nth * subgradient
        weights = projection_step(weights, lam)
    return weights

def kernelized_pegasos(X, y, lam):
    T = 1000

    #alpha counts how many times we've used that example so far
    alpha = np.zeros(X.shape[0])
    weights = np.zeros(X.shape[1])
    for t in range(1, T):
        nth = 1 / (lam*t)
        rand_idx = random.randrange(0, X.shape[0])
        summation = 0
        for j in range(0, X.shape[0]):
                kernel_func = kernel(X[rand_idx], X[j])
                summation += alpha[j] * y[rand_idx]* kernel_func
        if y[rand_idx] * nth * summation < 1:
            alpha[rand_idx] += 1

    for idx in range (0, X.shape[0]):
        weights += alpha[idx] * y[idx] * X[idx]

    return weights


def test_svm(X, weights):
    y_predict = np.array([])

 #   for w in range(len(weights)):
 #       if weights[w] == -0:
 #           weights[w] += 0
    for i in range(X.shape[0]):
        yp = np.sign(np.dot((X[i]), weights))
        y_predict = np.append(y_predict, yp)
    print(y_predict)
    return y_predict

"""
def test_kernelized(X, alpha):
    y_predict = np.array([])
    for i in range(len(y_test[:100])):
        for j in range(X.shape[0]):
            prediction += alpha[j] * y_train[j] * kernel(X_train[j], X_test[j])
        if prediction > 0:
            y_predict = np.append(y_predict, 1)
        else:
            y_predict = np.append(y_predict, -1)
    return None
"""


