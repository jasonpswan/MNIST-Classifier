import numpy as np

data_path = "./"

train_data = np.loadtxt(data_path + "mnist_train.csv",
        delimiter=",")
        
test_data = np.loadtxt(data_path + "mnist_test.csv",
        delimiter=",")