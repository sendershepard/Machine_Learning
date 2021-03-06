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
    """ This class will build the neural network with one hidden layer and ten
        outputs to classify each number from 0 to 9. The sizes of the neural
        network are held in an array delcaring the size of the input (i.e. 784),
        the size of the hidden neurons (e.g. 100), and the alpha fixed to .1.
        The network is pre-initialized with sizes [784, 100, 10] and an alpha
        learning rate of .1. """

    def __init__(self, sizes=[784, 100, 10], alpha=.1):
        """ Initializing the number of network layers, number of input nodes
            and number of hidden neurons and number of output neurons. """
        self = self
        self.num_layers = len(sizes) #Creating network layers, ie [784,20,10]
        self.num_input = sizes[0] 
        self.num_hidden = sizes[1]
        self.num_output = sizes[2]
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

        self.hidden_weights = np.random.uniform(-0.5, 0.5, size=(self.num_hidden, inputs))#.randn(self.num_hidden, inputs)#(5, 784)
        self.output_weights = np.random.uniform(-0.5, 0.5, size=(self.num_output, self.num_hidden))#randn(self.num_output, self.num_hidden)
        self.hidden_bias = np.ones((self.num_hidden, 1))
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

        
    def add_bias(self, data): #This function is not used, but helpful to have.
        """ This function will modify the data by adding a column of number
            1 to the data serving as a bias for our classifier. """
        samples = data.shape[0]
        onez = np.ones((samples, 1))

        return np.hstack((onez, data)) #Column-wise/horizontal array stacking
    

    def sigmoid(self, z): #Activation function for Network Classification
        """ This is a sigmoid neuron where the data point z= w*x + bias and
            so when the data is very large and positive it returns
            approximately 1, and when the data is very large and negative it
            returns approximately 0. """
        return (1.0 / (1.0 + np.exp(-z)))
    

    def sigmoid_prime(self, z):
        """ The sigmoid prime function calculates the derivative of the sigmoid
            or activation function. This function is used to compute the gradient
            descent using a back-propagation method. """
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))


    def feed_forward_prediction(self, data_points):
        """ In a feed forward perceptron, each dimension is multiplied
            by the layer's weights and the data points and the results are
            added together to the bias. We call our activation function
            using the sigmoid activation. If the total is greater than the
            threshold the neuron fires close to 1, else close to 0. """
        data_points = np.array(data_points, ndmin=2).T
        #a is the vector of activations of the second (hidden) layer of neurons
        h = self.sigmoid(np.dot(self.hidden_weights, data_points) +
                         self.hidden_bias)
        
        #o is the vector of activations of the output layer of neurons
        o = self.sigmoid(np.dot(self.output_weights, h) +
                         self.output_bias)

        return h, o #Important to save hidden/output values for back-prop algo


    def back_propagation(self, features, labels):
        """ This function is to update the weights using Stochastic Gradient
            Descent. The weights are updated with the gradient calculations
            from the feed forward prediction and then passed back using the
            back propagation algorithm. """
        hidden_output, network_output = self.feed_forward_prediction(features)#(5,1),(10,1)
        label = np.array(labels, ndmin=2).T #Avoiding matrix multiplication bugs
        fts = np.array(features, ndmin=2)
        output_errors = label - network_output #(10,1)
        hidden_errors = np.dot(self.output_weights.T, output_errors)#(10,5).T(10,1)=(5,1)

        """SDG: Updating the output weights. """
        delta_output = output_errors * self.sigmoid_prime(network_output)#(10,1)
        self.output_weights += self.alpha * np.dot(delta_output, hidden_output.T)

        """SDG: Updating the hidden weights. """
        delta_hidden = hidden_errors * self.sigmoid_prime(hidden_output) 
        self.hidden_weights += self.alpha * np.dot(delta_hidden, fts) #(5,1)(1,784)=(5,784)      

        self.alpha /= (1 + .0001*1) #Learning rate decay to reduce over-fitting
        
        return hidden_output, network_output, hidden_errors, output_errors


    def prediction(self, features):
        """ Simple prediction function that returns max index of the output. """
        output = self.feed_forward_prediction(features)[1] #Network output(10,1)

        return np.argmax(output, axis=0), np.max(output) #Index & percentage

    
    def test(self, features, label):
        """ Simple function to test to test the accuracy of a single data. """
        prediction, percentage =  self.prediction(features)
        correct = (prediction[0] == label) * 1
        
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


###########Training of Neural Network###########################
m = mnist()
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

###########Experiment 1#########################################
#For experiment 1, train one networks, using 20, 50, and 100
#hidden units in the hidden layer using .1 for learning rate 
################################################################
X_train1 = X_train
y_train1 = y_train
labels1 = m.init_labels(y_train)

###########Experiment 2#########################################
#For experiment 2, train two networks, using respectively
#one half and one quarter of the training examples for training
################################################################
#X_train1 = X_train[:5000] #X_Train has a size of 7,500 data points
#X_train2 = X_train[5000:7500]
#y_train1 = y_train[:5000]
#y_train2 = y_train[5000:7500]
#labels1 = m.init_labels(y_train1)
#labels2 = m.init_labels(y_train2)


for e in range(epochs):
    print("Epoch: ", e + 1)

    #Training two different sets on the same network using train1 & train2
    for i in range(len(X_train1)): m.back_propagation(X_train1[i], labels1[i])
    #for i in range(len(X_train2)): m.back_propagation(X_train2[i], labels2[i])

    train_c, train_w = m.evaluate_network(X_train1, y_train1)
    training_accuracy1 = train_c / ( train_c + train_w)
    print("Network's training1 accuracy: ", training_accuracy1)

    #train_c, train_w = m.evaluate_network(X_train2, y_train2)
    #training_accuracy2 = train_c / ( train_c + train_w)
    #print("Network's training2 accuracy: ", training_accuracy2)

    test_c, test_w = m.evaluate_network(X_test, y_test)
    testing_accuracy = test_c / ( test_c + test_w)
    print("**Network's testing accuracy: ", testing_accuracy)

    era[e + 1] = {"Training Accuracy1": training_accuracy1,
                  #"Training Accuracy2": training_accuracy2,
                  "Testing Accuracy": testing_accuracy}

print("\nConfusion Matrix for train1:\n", m.confusion_matrix(X_train1, y_train1))
#print("\nConfusion Matrix for train2:\n", m.confusion_matrix(X_train2, y_train2))

train1 = []
#train2 = []
test = []

print("=" * 31 + "Summary" + "=" * 31)
for i, acc in era.items():
    print(i, acc)
    train1.append(acc['Training Accuracy1'])
    #train2.append(acc['Training Accuracy2'])
    test.append(acc['Testing Accuracy'])
print("=" * 69)

""" Generating a line graph to measure the network's level of accuracy. """
plt.plot(train1, 'b', test, 'r')#train2, 'm', test, 'r')
plt.figure(1)
plt.legend(['train1', 'test'])#,'train2', 'test'])
plt.title("MNIST Challenge")
plt.ylabel("Accuracy Percentage")
plt.xlabel("Epochs")
#plt.savefig("results.png")
plt.show()

"""
End of Experiment 1 and 2
"""
