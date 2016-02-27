<<<<<<< HEAD
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    #puts all of the training data into a 60000x784 array and puts the true labels into a 60000 length array
    for i in range(10):
        trainx = mat['train'+str(i)]
        if len(train_data) == 0:
            train_data = trainx
        else:
            train_data = np.vstack((train_data, trainx))
        train_label = np.hstack((train_label, np.full(len(trainx),0,dtype=int)))
    print len(train_data)
    print len(train_label)
    print mat.keys()

    #extracts 10000 random entries from the training data and puts it into the validation data
    for i in range(10):
        index = np.random.randint(0, len(train_label))
        if len(validation_data) == 0:
            validation_data = train_data[index, :]
        else:
            validation_data = np.vstack((validation_data, train_data[index,:]))
        validation_label = np.hstack((validation_label, train_label[index]))
        train_data = np.delete(train_data, index, 0)
        train_label = np.delete(train_label, index)
    #puts all of the test data into a 60000x784 array and puts the true labels into a 60000 length array
    for i in range(10):
        testx = mat['test'+str(i)]
        if len(test_data) == 0:
            test_data = testx
        else:
            test_data = np.vstack((test_data, testx))
        test_label = np.hstack((test_label, np.full(len(testx),0,dtype=int)))

    print validation_data.shape
    print train_data.shape
    print len(train_label)
    print len(test_label)
    print test_data.shape
    #Pick a reasonable size for validation data



    return train_data, train_label, validation_data, validation_label, test_data, test_label


preprocess()
=======
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    print len(mat)

    #Pick a reasonable size for validation data


    #Your code here
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    return train_data, train_label, validation_data, validation_label, test_data, test_label


preprocess()
>>>>>>> origin/master
