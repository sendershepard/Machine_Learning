""" Sender Shepard """
""" Implementing Naive Bayes Classifiers
    based on Gaussian probability density function/distribution.
    The training and test files use the text files in the UCI datasets directory.
    Using this format, the data from the last column is the class labels, and
    hence it is not used to train as it is not a feature but a label. """

""" Files used 'yeast_training.txt', 'yeast_test.txt' """

""" Training output: this program models a P(x | class) as a Gaussian SEPARATELY
    for each dimension of the data. """

""" Classification output: each test object contains object ID, predicted class,
    probability of the predicted class given the data, true class, and accuracy. """

""" Accuracy output: the last output line prints the classification accuracy. """
import numpy as np
import math

class Gauss(object):
    def __init__(self):
        self.mean_dictionary = {} 
        self.variance_dictionary = {}
        self.unique_labels = []
        self.accur = 0
    
    
    def load_data(self, path="yeast_training.txt"):
        """ This function will load the data file, whether it be the training
            or the test set based on the argument pssed in path.
            This function will take the loaded data and will transform it
            by first splitting and separating data contents by " " and then
            it will create a new list without the empty spaces and finally
            it will transform the strings into floating point numbers. """
        data = []
        
        try:
            with open(path) as file:
                for row in file:
                    row = row.split(" ") #Separates contents if there is a space
                    row = list(filter(None, row)) #Removes the empty spaces
                    row = list(map(float, row)) #Transforms strings into floats
                    data.append(row)
            print("Loading of the ", path, " successful!")

        except:
            raise Exception("Load Failed")

        return data #np.asarray(data)


    def translate_traindata(self, data):
        """ This function will createa  dictionary with a label as a key, and
            the data that follows is an array of the data. If the label already
            exists, then add to the label an array within an array of data. """
        data_dict = {}

        for line in data:
            if line[-1] in data_dict: #if the last element/label exists already
                data_dict[line[-1]].append(line[:-1]) 
            else:
                self.unique_labels.append(line[-1]) #The last item is the label
                data_dict[line[-1]] = [line[:-1]] #Dict label: data except last

        return data_dict


    def gaussian_training(self, data): #, label):
        """ This function calculates the data's labels mean and variance, . """
        self.unique_labels = sorted(list(map(int, self.unique_labels)))

        for label in self.unique_labels:
            self.label_calculation(label, data) #Calculate mean & variance dicts

        probabilities = self.class_prob(self.unique_labels, len(data), data)

        return probabilities


    def label_calculation(self, label, data):
        """ This function will go through all of the data and it will calculate
            for each label, the mean and the  variance. For each label, there
            will be 8 elements/attributes for which the mean and variance is
            then saved in a dictionary data structure. """
        
        for col in range(len(data[label][0])): #Go through all 8 data elements
            mean = self.find_mean(data[label], col)
            var = self.find_variance(data[label], col, mean)

            if var < 0.0001:
                var = 0.0001 #Gaussian variance is NEVER smaller than 0.0001

            if label in self.mean_dictionary: #If the label already exists
                self.mean_dictionary[label].append(mean)
                self.variance_dictionary[label].append(var)
            else: #Else, create a new dictionary label and add a list
                self.mean_dictionary[label] = [mean]
                self.variance_dictionary[label] = [var]

            self.training_output(label, col, mean, var)#Output of training       

        
    def find_mean(self, data, column):
        """ This function will return the mean of the gaussian distribution
            based on the data's dimensions. Parameters µ corresponds to the
            mean of the distribution. That is Mu is the location parameter.
            The mean is obtained by summing all the values and dividing them
            the the number of values. sum(x)/n == μ"""
        total = 0
        
        for item in data:
            total += item[column] #Add all items in the same column in label

        return (total / len(data)) #Returns the calculated mean


    def find_variance(self, data, column, mean):
        """ This function will return the variance which is sigma^2
            and corresponds to the variance of the distribution.
            To calculate the variance, subtract the mean and square then
            square the result: sum((x−μ)^2)/n == σ^2"""
        total = 0

        for item in data:
            total += math.pow((item[column] - mean), 2)

        return (total / len(data))


    def class_prob(self, clabels, num_samples, data):
        """ This function will return a dictionary that holds the probability
            for each class of labels. """
        probabilities = {}

        for label in clabels:
            probabilities[label] = len(data[label]) / num_samples

        return probabilities

        
    def gaussian_distribution(self, data_point, mean, variance):
        """ The Gaussian PDF or normal distribution curve is used to calculate
            the probability distribution over measurement errors and provides
            a good approximation of the true distribution. The Gaussian
            formula is given by: f(x) = 1 / σ√2π * (e(−(x−μ)^2 / (2*σ^2))) """
        e_power = (((-1.0) * math.pow((data_point - mean), 2) /
                   (2 * variance)))
        e_log = math.pow(math.e, e_power) #e = −(x−μ)^2 / (2*σ^2)

        return e_log / (math.sqrt(variance * 2 * math.pi)) #e/σ√2π
    

    def test_data(self, path, class_prob):
        """ This function will load the test file and classify each row of the
            test file using the probability of classes that was derived from
            the training phase. It will then output the classification accuracy
            on the test data. """
        test_data = self.load_data(path)
        object_ID = 0
        
        for row in test_data:
            object_ID += 1
            self.classification_phase(row, class_prob, object_ID)

        self.accuracy_output(len(test_data))


    def classification_phase(self, data, class_prob, object_ID):
        """ This function shall classify the test data by calculating the
            Gaussian or probability density function and multiplying it by
            the class probabilty derived from the training phase.
            This function shall output the classification of the test data. """
        max_value = -1
        accur = 0
        pdf = self.data_classification(data)
        
        for label in pdf:
            pdf[label] *= class_prob[label]

        density_sum = sum(pdf.values())

        for label in pdf:
            pdf[label] /= density_sum
            
            if max_value < pdf[label]:
                max_value = pdf[label]
                p_class = label
        
        if p_class == data[-1]: #If the predicted class matches the label
            self.accur += 1
            accur = 1

        self.classification_output(object_ID, p_class, max_value, data, accur)

        return pdf 


    def data_classification(self, data):
        """ This class will classify the data's probability distribution
            by calling the Gaussian or probability density function, also
            known as the normal distribution. """
        pdf = {}
        
        for label in self.unique_labels:
            for col in range(len(data) - 1): #Avoiding the label column
                distribution = self.gaussian_distribution(data[col],
                                            self.mean_dictionary[label][col],
                                            self.variance_dictionary[label][col])
                if label in pdf:
                    pdf[label] *= distribution
                else:
                    pdf[label] = distribution
                    
        return pdf

        
    def training_output(self, label, column, mean, variance):
        """ Function that prints model P(x | class) as a Gaussian for each
            dimension of the data. Let the output of the training phase be:
            Class %d, attribute %d, mean = %.2f, std = %.2f. """
        print("Class %d, attribute %d, mean = %.2f, std = %.2f " % (
            label, column + 1, mean, math.sqrt(variance)))


    def classification_output(self, ID, predicted, probability, ctrue, accur):
        """ The output of the classification phase shall follow the sequence: 
            ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f
            Where the object ID is the line number of the object as it appears
            on the test file, the predicted class, the probability of the class,
            the true class, and the accuracy. """
        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f"
              % (ID, predicted, probability, ctrue[-1], accur))

        
    def accuracy_output(self, num_samples): 
        """ Function shall print the overall classification accuracy, which is
            defined as the average of the classification accuracies.  The
            classification accuracy shall follow this format:
            classification accuracy=%6.4f"""
        print("Classification Accurracy = %6.4f" % (self.accur / num_samples))


class naive_bayes(object):
    """ Naive Bayes Classifier"""
    def __init__(self, train_path, test_path):
        """ Initializing the path of the training and test files"""
        self.train_path = train_path
        self.test_path = test_path


    def run(self):
        """ This function will train the Naive Bayes Classifier on the training
            file and then classify the test file based on the probability
            density function or Gaussian distribution. """
        g = Gauss()
        l = g.load_data(self.train_path)
        d = g.translate_traindata(l)
        gt = g.gaussian_training(d)
        y = g.class_prob(g.unique_labels, len(d), d)
        yt = g.load_data(self.test_path)
        x = g.test_data(self.test_path, y)


""" Main """
if __name__ == '__main__':
    """
    Instructions, open the .py executable, then enter the name of the training file
    followed by a space and the name of the test file, such as:
    yeast_training.txt yeast_test.txt
    """
    user_input = input().split()
    #main = naive_bayes(user_input[0], user_input[1])
    main = naive_bayes('yeast_training.txt', 'yeast_test.txt')
    main.run()
    input()

