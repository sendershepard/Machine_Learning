""" Sender Shepard """
""" Implementing K-Means Unsupervised Classifier based on clustering.
The training and test files use the text files in the UCI datasets directory.
Using this format, the data from the last column is the class labels, and
hence it is not used to train as it is not a feature but a label. """

""" Training output: the program will run for a specified number of trials, in
this case 5, and will show the center's separation as the centers converge
during the training phase. When centers converge the next trial is then viewed
in the training output. """

""" Classification output: for each of the 5 trials, the average mean-square
error, mean-square-separation, and mean entropy (using the class labels) of
are displayed in the resulting clusteringon the training data. The best trial
is picked for testing classification. """

""" Accuracy output: line prints the testing classification accuracy using the
clustering of the best trial from trainig. """

""" The resulting clustering centers are visualized using the training clusters
centers.  That is, for each of the 10 cluster centers, the cluster center’s
attributes were used to draw the corresponding digit on an 8 x 8 grid.
The grid of 8x8 is fairly small for the high resolution monitors so a zoom
was needed to better-view the digit drawn in a grayscale value. These digits
are saved in the current directory for ease of viewing instead of displaying
10 separate windows. """

import numpy as np
import math
from PIL import Image


class k_means_algorithm(object):
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.unique_labels = []

    def load_file(self, path='optdigits.train'):
        """ This function will load the data file, whether it be the training
            or the test set based on the argument pssed in path.
            This function will take the loaded data and will transform it
            by first splitting and separating data contents by "," and then
            it will create a new list without the empty spaces and finally
            it will transform the strings into floating point numbers. """
        features = [] 
        labels = []
        
        try:
            with open('./optdigits/' + path) as file:
                for row in file:
                    row = row.split(',')
                    row = list(filter(None, row))
                    row = list(map(float, row))
                    features.append(row)
            print("Loading of the ", path, " successful!")

        except:
            raise Exception("Load Failed")

        return features

    def translate_file(self, file):
        """ This function shall translate the data into an easily readible
            format. It will also identify the unique labels of all the data. """
        data = []
        labels = []

        for line in file:
            data.append(line[:-1])
            labels.append(line[-1])
            
            if line[-1] not in labels[:-1]:
                self.unique_labels.append(line[-1])

        self.unique_labels = sorted(list(map(int, self.unique_labels)))

        return np.asarray(data), (np.asarray(labels).astype(int))

    def initialize_centroids(self, data):
        """ Randomly assign centroids. This function will create a random seed
            from which it will initialize data based on feature specs: 64
            elements and each element ranges from 0-16. Returns a matrix of
            n_clusters by 64 elements, each ranging from 0-16. """
        random_center = np.random.randint(min(data[0]), max(data[0]),
                                          np.shape(data)[1])
        
        return [random_center for i in range(self.n_clusters)]

    def euclidean(self, data, center, distance=0):
        """ The Euclidean distance is meassured by the square root of the sum
            of the difference between x and y. So the Euclidean distance is
            expressed as the sqrt(sum of (Xi - Yi)^2) or sqrt(∑(Xi − Yi)^2). """
        for i in range(len(data)):
            distance += math.pow((data[i] - center[i]), 2)

        return np.sqrt(distance)      

    def closest_euclidean(self, point, centroids):
        """ Returns index of the closest data point to center. If there are
            multiple distances that are minimum, then return a random index. """
        point_distance = []

        for centroid in centroids:
            point_distance.append(self.euclidean(point, centroid))

        min_dist, min_index = [], np.argmin(point_distance)
        
        for m in range(len(point_distance)):
            if (point_distance[m] - point_distance[min_index]) < (10 ** -10):
                min_dist.append(m)

        return np.random.choice(min_dist)

    def centroid_convergance(self, old_centers, new_centers, centers_distance=0):
        """ This function shall return True when the centroids have stopped
            moving and hence the old and new centroids have converged. """
        for i in range(len(old_centers)):
            centers_distance += np.abs(old_centers[i] - new_centers[i])

        print("Centroid separation: ", np.sum(centers_distance))
        if np.sum(centers_distance) < 0.1: 
            return True #If centers have converged

        return False

    def compute_ss_error(self, data, centers, clusters, sse=0):
        """ This function returns the lowest sum of squared error. This error
            is comuted by taking the sum of all the distances between the data
            point and the centroid from cluster i, squared. ∑ d(p, Ci)^2 """
        for i in range(self.n_clusters):
            for point in clusters[i]:
                sse += self.euclidean(data[point], centers[i]) ** 2
                
        return sse

    def sum_squared_separation(self, centers, sss=0):
        """ This function shall calculate the sum squared separation of the
            clustering where all pairs are distinct Mi, Mj. The formula is
            given by ∑ euclidean_distance(Mi, Mj)^2. """
        def unordered_pairs():
            """ Helper function that provides a tuple combination of unordered
                mairs where x, y are not equal (x != y) used in SSS. """
            pairs = []
            for x in range(9):
                y = x+1
                while y < 10:
                    pairs.append((x, y))
                    y +=1
            return pairs

        unordered_pairs = unordered_pairs()
        
        for pair in unordered_pairs:
            sss += self.euclidean(centers[pair[0]], centers[pair[1]]) ** 2

        return sss

    def entropy(self, cluster, labels):
        """ This function will find the entropy of the labels per cluster. In
            order to find the entropy of a cluster we use Shannon's entropy
            defined as: H(Ci) = -∑ P(Xi,j) * log2(P(Xi,j)) where x is an element
            which represents cluster i and label class j.
            P(Xi, j) is the function of probablity that a data from cluster Ci
            belongs to j. Therefore, P(Xi,j) = Xi/j, is computerd by how many
            labels are from Xi in class j divided by all the existing labels
            of class j. """
        entropy = 0

        label_count = [0 for i in range(self.n_clusters)]
        for point in cluster:
            label_count[labels[point]] += 1
        
        for i in range(self.n_clusters): #∑ P(Xi,j) * log2(P(Xi,j))
            Xi = label_count[i]
            j = len(cluster)

            if label_count[i] == 0:
                entropy_summation = 0
            else:
                entropy_summation = ((Xi/j) * math.log((Xi/j), 2))
            entropy += entropy_summation

        return -entropy #H(Ci) = -∑ entropy
        
    def mean_entropy(self, clusters, labels):
        """ This function shall compute the mean entropy of a clustering which
            is defined as the average entropy of all the clusters. The formula
            is as follows: H(Omega) = ∑ Entropy(Ci) * Xi/N where Omega is the
            set of clusters. Xi is the number of points in cluster i,
            and divided by N, the total number of label points. We want to
            minimize mean entroy. """
        m_e = 0

        for i in range(self.n_clusters):
            num_Ci = len(clusters[i])
            total_labels = len(labels)
            m_e += (self.entropy(clusters[i], labels) * (num_Ci/total_labels))

        return m_e #∑ (Entropy(Ci) * ( Xi/N ))

    def kmeans_training(self, train_data, train_labels, num_trials=5):
        """ This function shall train using the K-Means algorithm and return
            a dictionary with the trial's average mean-square-error,
            mean-square-separation, and mean entropy . """
        trial_results = {}

        for trial in range(num_trials):
            print('\n', '*'*10, "K-Means training trial", trial + 1, '*'*10)
            print("Initializing random centroids...")
            centers = self.initialize_centroids(train_data)

            convergance = False
            while convergance is False: #Continue until centers converge
                closest_idx = []
                clusters = [[] for i in range(len(centers))]

                #1.- Calculating the closest centers for all points
                for point in train_data:
                    closest_idx.append(self.closest_euclidean(point,centers))

                for i in range(len(closest_idx)):
                    clusters[closest_idx[i]].append(i)

                centroids = [] #(10x64)
                #2.- Calculating the centroid for each cluster
                for cluster in clusters:

                    mean_vector = np.array([0.0 for i in range(64)])
                    for i in range(len(cluster)): #64 elems per point
                        mean_vector += np.array((train_data[cluster[i]]))

                    if len(cluster) > 0:
                        mean_vector /= float(len(cluster))

                    centroids.append(mean_vector) #(10x64)

                #3.- Updating centers and checking for convergance
                old = centers
                centers = centroids
                convergance = self.centroid_convergance(old, centers)

            sse = self.compute_ss_error(train_data, centers, clusters)
            sss = self.sum_squared_separation(centers)
            mean_entropy = self.mean_entropy(clusters, train_labels)

            trial_results[trial] = [sse, sss, mean_entropy,
                                    centers, closest_idx, clusters]
        return trial_results
                
    def training_results(self, trial_results):
        """ This is a helper function that prints the results of the K-Means
            trials. The report shall print the average mean_square_error,
            mean_square_separation, and mean_entropy from the training data
            clusters. This function will return the trial that
            yields the smallest average Mean Square Error. """
        best_trial = 0
        num_trials = len(trial_results)

        for i in range(num_trials):
            print("\nTrial", i + 1, "Results:")
            print("*"*3, "The Average Mean Square Error:\t", trial_results[i][0])
            print("*"*3, "The Mean Square Separation:\t\t", trial_results[i][1])
            print("*"*3, "The Mean Entropy:\t\t\t", trial_results[i][2])

        for trial in range(1, num_trials):
            if trial_results[trial][0] < trial_results[best_trial][0]:
                best_trial = trial
        print("\nBest trial was trial number:", best_trial + 1)

        return trial_results[best_trial]

    def most_frequent_class(self, cluster, labels):
        """ This class shall associate each cluster center with the most
            frequent class it containsin the training data.
            If there is a tie for most frequent class, break at random."""
        tied_classes = []
        cluster_count = [0 for i in range(self.n_clusters)]

        for data_point in cluster:
            cluster_count[labels[data_point]] += 1

        most_frequent = max(cluster_count) #The most frequent class and index
        mf_index = cluster_count.index(most_frequent)

        if cluster_count.count(most_frequent) == 1:
            return mf_index #If there are no ties of most popular class

        for i in range(len(cluster_count)):
            if cluster_count[i] == most_frequent:
                tied_classes.append(i)

        return np.random.choice(tied_classes) #If tie, break at random

    def frequent_classes(self, clusters, labels):
        """ This function shall call the most_frequent_class function in order
            to get all of the cluster's most frequent classes. """
        most_freq_classes = []

        for cluster in clusters:
            most_freq_classes.append(self.most_frequent_class(cluster, labels))

        return most_freq_classes

    def class_classification(self, test_data, centers, cluster_class):
        """ This function shall assign each class to a cluster based on how
            frequent that class appears in the training data. It shall return
            the classifications for each test data. """
        closest = self.closest_euclidean(test_data, centers)

        return cluster_class[closest]

    def confusion_matrix(self, test_labels, class_classifications):
        """ This class shall create the confusion matrix. If the label and the
            classification match, then add one to the element in the array which
            has been previously initialized to all zero. """
        confusion_matrix = [[0 for k in range(self.n_clusters)]
                            for i in range(self.n_clusters)]

        for label, classif in zip(test_labels, class_classifications):
            confusion_matrix[label][classif] += 1

        return confusion_matrix

    def kmeans_accuracy(self, confusion_matrix):
        """ This function shall calculate the accuracy on the test data by
            going through the confusion matrix's results. """
        conf_matrix = np.asarray(confusion_matrix)

        return (float(np.sum(np.diagonal(conf_matrix))) / np.sum(conf_matrix))
 
    def display_confusion(self, confusion_matrix):
        """ This function shall display the confusion matryx. """
        for x in range(len(confusion_matrix)):
            print(*confusion_matrix[x], sep="\t")
            
    def draw(self, name, cluster_number, center):
        """ Visualize the resulting cluster centers.  That is, for each of the
            10 cluster centers, use the cluster center’s attributes to draw the
            corresponding digit on an 8 x 8 grid.   To do this, each value in the
            cluster center’s feature vector is interpreted as agrayscale value
            for its associated pix. """
        def pixel_value(val):
            """ Helper function to provide a pixel value. """
            val = int(np.floor(val))

            return val * 16
        
        img = Image.new('L', (8, 8), "black")
        cent = np.array(center).reshape(8, 8)

        for i in range(img.size[0]):
            for k in range(img.size[0]):
                img.putpixel((k, i), pixel_value(int(cent[i][k])))

        #img.show()
        name = name + str(cluster_number) + ".jpg"
        img.save(name)



class k_means(object):
    """ K-Means Main """
    def __init__(self, train_path, test_path):
        """ Initializing the path of the training and test files"""
        self.train_path = train_path
        self.test_path = test_path

    def run(self, num_clusters, n_trials):
        """ This function will train the K-Means Classifier on the training
            file and then classify the test file based on the Clustering of the
            most frequent classes. """
        k = k_means_algorithm(num_clusters)
        
        train_file = k.load_file(self.train_path)
        test_file = k.load_file(self.test_path)
        
        train_points, train_labels = k.translate_file(train_file)
        test_points, test_labels = k.translate_file(test_file)

        centers = k.initialize_centroids(train_points)

        
        kmeans = k.kmeans_training(train_points, train_labels, n_trials)
        best_trial = k.training_results(kmeans)
        trained_clusters = best_trial[5]
        trained_centers = best_trial[3]
        
        frequent_clusters = k.frequent_classes(trained_clusters, train_labels)

        classified_labels = [k.class_classification(
            point, trained_centers, frequent_clusters)for point in test_points]


        confusion_matrix = k.confusion_matrix(test_labels, classified_labels)
        accuracy = k.kmeans_accuracy(confusion_matrix)

        print("With accuracy: ", accuracy)
        print("\nConfusion Matrix:")
        k.display_confusion(confusion_matrix)

        #The vizualisation reports are saved in the same directory as the .py
        for i in range(len(trained_centers)):
            k.draw("Resulting_Center_", i+1, trained_centers[i])        


""" Main """
if __name__ == '__main__':
    num_trials = 5 
    num_clusters = 10
    main = k_means('optdigits.train', 'optdigits.test')
    main.run(num_clusters, num_trials)
    #input()
