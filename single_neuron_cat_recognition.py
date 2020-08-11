# import libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
import skimage.transform as tf

%matplotlib inline 


# load training data
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    classes = np.array(test_dataset["list_classes"][:]) # 'non-cat', and 'cat'
    
    train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])
    
    #print (test_set_y_orig.shape)
    #print (test_set_y_orig[0])
    #print (classes[1])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# load data into containers
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#print (classes[1])
#print (train_set_y.shape)
#print (train_set_y[:,100])

# a = np.zeros ((9,9))
# a[1,1] = 1
# print (a[1,:])


# confirm the variables' shapes
print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))


# confirm the sample size of train and test set, confirm sample image pixel
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

print("train set sample size: m_train = " + str(m_train))
print("test set sample size: m_test = " + str(m_test))
print("pixel per edge: num_px = " + str(num_px))


# flatten and transpose the orignal vectors
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))


# standarise the data (remap to 0~1)

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

print(test_set_x.shape)
# print(test_set_x)


"""
Sigmoid function
Parameters:
    z: a parameter or np array
Returns:
    s: sigmoid result, ranged in [0,1]
"""
def sigmoid(z):
    
    s = 1 / (1 + np.exp(- z))
    
    return s


"""
Initialize the weight w, and bias b
Parameters:
    dim: the dimension of vector w
Returns:
    w: weight
    b: bias
"""
def initialize_with_zeros(dim):
    
    w = np.zeros((dim,1))
    b = 0
    #print(w.shape)
    
    return w, b


"""
Forward and backward propagate
Parameters:
    w: weight array
    b: bias
    X: characteristics array
    Y: labels array
Returns:
    cost: cost
    grads: gradients
        dw: weight gradient
        db: bias gradient
"""
def propagate(w, b, X, Y):
    
    # get the sample size
    m = X.shape[1]
    
    # forward propagate
    Z = np.dot(w.T, X) + b # Z.shape = (1, sample size)
    A = sigmoid(Z) # A.shape = (1, sample size)
    cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m
    
    # backward propagate
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    
    grads = {'dw': dw,
             'db': db}
    
    return grads, cost


"""
 Gradient descent
 Parameters:
      w: weights array, (12288, 1)
      b: bias
      X: characteristics array, shape=(12288, 209)
      Y: labels array, 0/1, 0=non-cat, 1=cat, shape=(1, 209)
      num_iterations: dedicated optimization iterations
      learning_rate: learning rate
      print_cost: print the cost value per 100 iterations if Ture
 Returns:
      params: optimized w and b
      costs: history cost values, record per 100 iteratioins
"""
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for num in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if(num % 100 == 0):
            costs.append(cost)
            if print_cost:
                print ("The cost after %i times of optimization is: %f" %(num, cost))
    
    params = {'w': w,
              'b': b}    
        
    return params, cost


"""
 Predict whether the image contain cat
 Parameters:
     w: weights array, shape=(12288, 1)
     b: bias
     X: characteristics array, shape=(12288, sample size)
 Returns:
     Y_prediction: prediction to each image

"""
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(m):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
            
    return Y_prediction


"""
 Neral network model function
 Parameters:
     X_train: train set, shape=(12288, 209)
     Y_train: labels of train set, shape=(1, 209)
     X_test: test set, shape=(12288, 50)
     Y_test: labels of test set, shape=(1, 50)
     num_iterations: number of train/optimize needed
     learning_rate: learning rate
     print_cost: print the cost values per 100 iterations if set to True
 Returns:
     d: predictions regard test and train sets, weight, bias, learning rate, number of iterations
"""
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize the parameters to be trained
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # train the parameters and get them
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    
    # predict the train & test image set w/ the trained weight and bias
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    # print the prediction accuracy
    print("The accuracy regarding train image set is: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("The accuracy regarding test image set is: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2001, learning_rate = 0.005, print_cost = True)