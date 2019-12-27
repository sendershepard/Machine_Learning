"""
    Sender Shepard

    Machine Learning
    The MNIST CHALLENGE using Neural Networks
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class mnist(object):
    """ This class will build the perceptron classifiers as ten
        outputs to classify each number from 0 to 9. The sizes of the neural
        network are held in an array delcaring the size of the input (i.e. 784),
        the size of output neurons (i.e. 10), and the alpha defaulted to .1.
        When initializing the mnist object, the alpha can be changed to 0.01 or
        0.001. """

    def __init__(self, sizes=[784, 10], alpha=.1):
        """ Initializing the number of network layers, number of input nodes
            and number of hidden neurons and number of output neurons. """
        self = self
        self.num_layers = len(sizes) #Creating network layers, ie [784, 10]
        self.num_input = sizes[0] 
        self.num_output = sizes[1]
        self.alpha = alpha #Learning rate


    def load_file(self, mnist_file='mnist_test.csv'):
        """ This function will load the mnist file, whether it be the training
            or the test set based on the argument pssed in mnist_file. """
        data = []
        
        try:
            with open(mnist_file) as f:
                read_file = csv.reader(f) #Creating an open stream to read 
                for row in read_file: #Reading each line to create an array
                    data.append(row)
                    #data.append(np.asarray(row,dtype=np.int8))
            print("Loading of the ", mnist_file, " successful!")
            
        except:
            raise Exception('Load failed')
        
        return np.asarray(data)


    def view_image(self, index, data):
        """ This function will help us visualize our digits. Since the data
            image is of a 28x28 pixles we create img with these dimensions.
            Furthermore, the data comes in as rows. """
        pixels = data[index].reshape((28, 28)) #28x28 pixles per image
        plt.imshow(pixels, cmap = 'gray')
        plt.show()

        
    def init_weights(self, data):
        """ This function will initialize the weights to a random number. The
            weights will be class attributes that will represent the hidden
            weights and the output weights. The hidden weights will represent
            the hidden layer in a fully connected Neural Network. """
        inputs = data.shape[1] #784 inputs/pixels

        self.output_weights = np.random.uniform(-0.5, 0.5, size=(self.num_output, inputs))#randn(self.num_output, self.num_hidden)
        self.output_bias = np.ones((self.num_output, 1))


    def init_labels(self, labels):
        """ Transforming labels into a binary representation within an array.
            So, position 0 represents 1 and the rest of the array is filled with
            0, for example. The function will take an array of any size and
            return it with the transformation of the so called hot_labels.
            That is to say, the index i, will be represented by 1 or 'hot'. """
        label_index = np.arange(10) #List of labels 0 through 9
        list_labels = (label_index == labels).astype(np.float)
        list_labels[list_labels == 0] = 0.01 #Value 0 prevents weight updates
        list_labels[list_labels == 1] = 0.99 #Value 1 prevents weight updates

        return list_labels #Returns an array of 10 where 0.01 is 0 and 0.99 is 1
    

    def perceptron_training(self, features, label):
        """ This function is to update the weights using Stochastic Gradient
            Descent. The weights are updated with the gradient calculations
            from the feed forward prediction and then passed back using the
            back propagation algorithm. """
        label = np.array(label, ndmin=2).T #Avoiding matrix multiplication bugs
        new = (self.alpha * (label - self.ff_prediction(features)) * features)
        self.output_weights +=  new

        self.alpha /= (1 + .0001 * 1) #Learning rate decay: reduce over-fitting
        
        return new


    def dot_prod(self, data):
        """ Returns the dot product of the weights and the data. It really is a
            matrix multiplication of a 1x784 dot 784x10 = 10x1 per data point.
            This is a helper function. """
        data = np.array(data, ndmin=2).T
        return (np.dot(self.output_weights, data) + self.output_bias)


    def ff_prediction(self, data):
        """ Will return the prediction of where there is a perceptron match.
            The prediction will be a 10x1 matrix with 1 where the product has
            been greater than 0. Else 0. If the total is greater than the
            threshold then the neuron fires 1,
            else 0."""
        return np.where(self.dot_prod(data) >= 0.0, 1, 0) #THE MAGIC


    def prediction(self, features):
        """ Simple prediction function that returns max index of the output. """
        output = self.ff_prediction(features) #Perceptron Output(10,1)

        return np.argmax(output, axis=0), np.max(output) #Index & percentage


    def test(self, features, label):
        """ Simple function to test to test the accuracy of a single data. """
        prediction, percentage =  self.prediction(features)
        correct = (prediction == label) * 1
        
        return prediction, int(label), int(correct[0]), percentage


    def evaluate_network(self, data, labels):
        """ This function will evaluation the neural network's accuracy by
            finding the prediction of the data, and determine if it was
            accurately predicted or not. Returns total corrects and wrongs. """
        corrects, wrongs = 0, 0

        for i in range(len(data)):
            prediction, label, correct, percent = self.test(data[i], labels[i])

            if correct:
                corrects += 1
            else:
                wrongs += 1

        return corrects, wrongs


    def confusion_matrix(self, data, labels):
        """ This function will create a confusion matrix for the trained network
            that will return a 10x10 array summarizing the resutls of the test
            set's. """
        c_matrix = np.zeros((10,10), int)
        
        for i in range(len(data)):
            prediction, label, correct, percent = self.test(data[i], labels[i])
            c_matrix[prediction, label] += 1
            
        return c_matrix


    def display_confusion(self, confusion_matrix):
        """ This function shall display the confusion matryx. """
        for x in range(len(confusion_matrix)):
            print(*confusion_matrix[x], sep="\t")


###########Training with Perceptron Rule############################

###########Experiment 1#############################################
#Train the perceptrons with learning rates: η = 0.001, 0.01, and 0.1
#Repeat for 50 epochs.
####################################################################
m = mnist(alpha=0.1) #learning rates: η = 0.001, 0.01, and 0.1
error = []
era = {}
epochs = 50
#data_set = m.load_file(mnist_file='mnist_train.csv') #Full set throws memory errors
data_set = m.load_file() #Testing set by default; full set throws memory errors 

""" Splitting the test data for testing and training purposes using a
    much smaller set to work with for now as full set throws memory errors. """
digit = np.asarray(data_set[:, 0:1], dtype='float')
features = np.asarray(data_set[:, 1:], dtype= 'float')*((0.99/255) + 0.01)

""" Train/Test Spliting for testing. """
X_train, X_test, y_train, y_test = train_test_split(features,
                                                digit, test_size=0.2,
                                                random_state=42)
m.init_weights(X_train)
labels = m.init_labels(y_train)


for e in range(epochs):
    print("Epoch: ", e + 1)

    for i in range(len(X_train)): m.perceptron_training(X_train[i], labels[i])
    
    train_c, train_w = m.evaluate_network(X_train, y_train)
    training_accuracy1 = train_c / ( train_c + train_w)
    print("Network's training accuracy: ", training_accuracy1)

    test_c, test_w = m.evaluate_network(X_test, y_test)
    testing_accuracy = test_c / ( test_c + test_w)
    print("**Network's testing accuracy: ", testing_accuracy)

    era[e + 1] = {"Training Accuracy": training_accuracy1,
                  "Testing Accuracy": testing_accuracy}


###########Printing Perceptron's Training Results ############################
print("\nConfusion Matrix:\n")#, m.confusion_matrix(X_train, y_train))
m.display_confusion(m.confusion_matrix(X_test, y_test))
                    
train1 = []
test = []

print("=" * 31 + "Summary" + "=" * 31)
for i, acc in era.items():
    print(i, acc)
    train1.append(acc['Training Accuracy'])
    test.append(acc['Testing Accuracy'])
print("=" * 69)

""" Generating a line graph to measure the network's level of accuracy. """
plt.plot(train1, 'b', test, 'r')
plt.figure(1)
plt.legend(['train1', 'test'])
plt.title("MNIST Challenge")
plt.ylabel("Accuracy Percentage")
plt.xlabel("Epochs")
#plt.savefig("results.png")
plt.show()

