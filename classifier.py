import numpy as np
import sys
import matplotlib.pyplot as plt
from os import system, name
from time import sleep

from perceptron import *

# Load the train and test data is the initial step here.

print ("Data is now being loaded. . . ")
image_pixels = 28 * 28
data_path = "./"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",delimiter=",")
print("Data has now been loaded . . . ")

# Part 1: Uses the train and test from the perceptron class in order to identify the target digit (7).

def problem_one():
   
    print("Part 1 is now being run.")
    
    target = 7
    p = Perceptron(28*28+1)
    p.print_details()
    
    print("Training has now begun . . . ")
    training_data = [np.append([1],d[1:]) for d in train_data]
    labels = [d[0]==target for d in train_data]
    p.train(training_data, labels)
    print("Training has now completed . . . ")
    
    print("Testing has now begun ...")
    testing_data = [np.append([1],d[1:]) for d in test_data]
    labels = [d[0]==target for d in test_data]
    p.test(testing_data, labels)
    
    print ("Part 1 is now complete.")

problem_one()









# Part 2: Update the perceptron to use Batch Learning

# So I wrote a small code called batch_training in the perceptron file.
# This will be used for training in place of the previous train code.
# The test code remains the same.

def problem_two():
    
    print("Part 2 is now being run.")
    
    target = 7
    q = Perceptron(28*28+1)
    q.print_details()
    
    print("Batch Training. . . ")
    training_data = [np.append([1],d[1:]) for d in train_data]
    labels = [d[0]==target for d in train_data]
    q.batch_training(training_data, labels)
    print("Batch Training complete. . . ")
    
    print("Testing...")
    testing_data = [np.append([1],d[1:]) for d in test_data]
    labels = [d[0]==target for d in test_data]
    q.test(testing_data, labels)
    
    print("Part 2 is now complete.")

problem_two()











# Part 3: Use multiple nodes to classify every handwritten digit.

# Previously we have had one target digit.
# We now wish to classify all 10 digits, which means we gots to have 10 nodes.
# Each node is pa, where a represents the target digit (p0 targets 0 etc.)

def problem_three():
    
    print("Part 3 is now being run.")
    
    p0 = Perceptron(28*28+1)
    p1 = Perceptron(28*28+1)
    p2 = Perceptron(28*28+1)
    p3 = Perceptron(28*28+1)
    p4 = Perceptron(28*28+1)
    p5 = Perceptron(28*28+1)
    p6 = Perceptron(28*28+1)
    p7 = Perceptron(28*28+1)
    p8 = Perceptron(28*28+1)
    p9 = Perceptron(28*28+1)
    
    nodes = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    
    print("Training has now begun . . . ")
    training_data = [np.append([1],d[1:]) for d in train_data]
    print("Training has now been completed!")
    
    
    # A for loop is used to iterate through the list of nodes above
    # This will train each node based on it's index in the list
    # The index matchs the digit following the p
    
    
    print("Assigning nodes digits to classify, sorry this takes a while . . . ")
    for i in range(len(nodes)):
        labels = [d[0]==i for d in train_data]
        nodes[i].train(training_data,labels)
    print("The nodes have been assigned digits to classify")
    
    # We are now ready to test the model again.
    # This isn't as easy as before, as we now have 10 target digits.
    
    print("Testing sequence has been initiated . . . ")
    
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    
    for t in test_data:

        for i in range(len(nodes)):
            labels = [t[0]==i]
            prediction = nodes[i].predict(t)
            if prediction == 0:
                if labels[0] == True:
                    fn = fn + 1
                continue
            else:
                if labels[0] == True:
                    tp = tp + 1
                    break
                else:
                    fp = fp + 1
                    break

    tn = len(test_data) - tp - fp - fn
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(" Accuracy:\t"+str(accuracy))

    precision = tp/(tp+fp)
    print(" Precision:\t"+str(precision))
    
    recall = tp/(tp+fn)
    print(" Recall:\t"+str(recall)) 
    
    print("Part 3 is now complete")
    
problem_three()






# Part 4: Update the perceptron to use the sigmoid activation function. 

# So now we gotta edit the predict function to be a predict sigmoid function.
# And we'll update the train function to use the sigmoid function.
# This will be done in the perceptron as it will literally just be changing the
# words predict & train here.

# I did also run it for part 3 problem but the code takes a lot longer to run.
# Hope you don't mind.
# I'll be using sigmoid and part 3 problem combined for visualisation in part 5 anyway.

def problem_four():
    
    # Due to how much quicker part 1 runs compared to part 3.
    # I'll run the test for identifying 7 using the sigmoid activation.
    # By editing the boundary (0.5 as a base) we can see an increase in all three measures
    # (Accuracy, Precision, Recall) when comparing sigmoid to basic activation.
    # Measured at the following cut offs: 0.25, 0.5, 0.75, 0.875
    # 0.875 gave the highest values for all three measures when comparing.
    
    print("Part 4 is now being run.")
    
    target = 7
    p = Perceptron(28*28+1)
    p.print_details()
    
    print("Training has now begun . . . ")
    training_data = [np.append([1],d[1:]) for d in train_data]
    labels = [d[0]==target for d in train_data]
    p.train_sigmoid(training_data, labels)
    print("Training has now completed . . . ")
    
    print("Testing has now begun ...")
    testing_data = [np.append([1],d[1:]) for d in test_data]
    labels = [d[0]==target for d in test_data]
    p.test(testing_data, labels)
    
    print ("Part 4 is now complete.")
    
problem_four()


# Part 5: Visualisation is the key

# This small function will display the weight images later.
# It forms the final part of Problem 5 below


def visualise_weights(p,i):
    
    vis = p.weights[1:].reshape((28, 28))
    plt.imshow(vis, cmap = 'viridis')
    plt.title(i)
    plt.show()

def problem_five():
    
    print("Part 5 is now being run.")
    
    p0 = Perceptron(28*28+1)
    p1 = Perceptron(28*28+1)
    p2 = Perceptron(28*28+1)
    p3 = Perceptron(28*28+1)
    p4 = Perceptron(28*28+1)
    p5 = Perceptron(28*28+1)
    p6 = Perceptron(28*28+1)
    p7 = Perceptron(28*28+1)
    p8 = Perceptron(28*28+1)
    p9 = Perceptron(28*28+1)
    
    nodes = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    
    print("Training has now begun . . . ")
    training_data = [np.append([1],d[1:]) for d in train_data]
    print("Training has now been completed!")
    
    
    # A for loop is used to iterate through the list of nodes above
    # This will train each node based on it's index in the list
    # The index matchs the digit following the p
    
    
    print("Assigning nodes digits to classify, sorry this takes a while . . . ")
    for i in range(len(nodes)):
        labels = [d[0]==i for d in train_data]
        nodes[i].train(training_data,labels)
    print("The nodes have been assigned digits to classify")
    
    for t in test_data[1:10]:
        data = t[1:].reshape((28, 28))
        for a in range(28):
            for b in range(28):
                if data[a][b]>0:
                    sys.stdout.write("#")
                else:
                    sys.stdout.write(".")
                sys.stdout.flush()
            sys.stdout.write("\n")
            
    for i in range(len(nodes)):
        visualise_weights(nodes[i],i)
    
problem_five()

