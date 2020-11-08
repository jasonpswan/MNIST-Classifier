import numpy as np

class Perceptron(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, max_iterations=20, learning_rate=0.1):
        self.no_inputs = no_inputs
        self.weights = np.ones(no_inputs) / no_inputs
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    #=======================================#
    # Prints the details of the perceptron. #
    #=======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    #=========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    #=========================================#
    def predict(self, inputs):
        activation = np.dot(inputs, self.weights)
        if activation > 0:
            return 1
        return 0

    #======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    #======================================#
    def train(self, training_data, labels):
        
        assert len(training_data) == len(labels)
        
        for i in range(self.max_iterations):
            for data, label in zip(training_data, labels):
                prediction = self.predict(data)
                self.weights = self.weights + self.learning_rate * (label - prediction) * data
        return

    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy of the perceptron. #
    #=========================================#    
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for data, label in zip(testing_data, labels):
            prediction = self.predict(data)
            
            if label == 1:
                if prediction == 1:
                    tn = tn + 1
                else:
                    fp = fp + 1
                
            else:
                if prediction == 1:
                    fn = fn + 1
                else:
                    tp = tp + 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision= tp / (tp + fp)
        recall = tp / (tp + fn)
    
        print("Accuracy:\t"+str(accuracy))
        print("Precision:\t"+str(precision))
        print("Recall:\t"+str(recall))
        
        
    #======================================#
    # Training using Batch Learning.       #
    #======================================#
    
    # The model is updated to include batch learning, as per part 2 of the assignment.
    
    def batch_training(self, training_data, labels):
        
        assert len(training_data) == len(labels)
        
        for i in range(self.max_iterations):
            weight_update = np.zeros(len(self.weights))
            for data, label in zip(training_data, labels):
                prediction = self.predict(data)
                weight_update = weight_update + self.learning_rate * (label - prediction) * data
            self.weights += weight_update/len(training_data)
        return
            
    
    #======================================#
    # Using the sigmoid activation         #
    # function                             #
    #======================================#
    
    # We now have to edit predict to use the sigmoid.
    # And edit train to use the predict sigmoid.
    
    def predict_sigmoid(self, inputs):
        x = np.dot(inputs, self.weights)
        sigmoid = 1/(1 + np.exp(-x))
        if sigmoid > 0.875:
            return 1
        return 0
    
    def train_sigmoid(self, training_data, labels):
        
        assert len(training_data) == len(labels)
        
        for i in range(self.max_iterations):
            for data, label in zip(training_data, labels):
                prediction = self.predict_sigmoid(data)
                self.weights = self.weights + self.learning_rate * (label - prediction) * data
        return
    
    