""" Sender Shepard """
""" Gaussian Mixture Model and K-Means Clusterings. """
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from matplotlib import style
from scipy.stats import multivariate_normal
style.use('fivethirtyeight')
#from scipy.stats import norm

""" K-Means Algorithm"""
class KMeans_Algorithm(object):
    """ This class will be the K-Means algorithm and
        establish the K-Means Algorithm. """
    def __init__(self, n_clusters, max_iterations=100, random_state=312):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        
        self.x, self.y = make_blobs(n_samples=500, centers=n_clusters,
                          cluster_std=0.60, random_state=0)

    def _initialize_centroids(self, data):
        """ Randomly assign centroids. """
        np.random.RandomState(random.randrange(1, 1001, 4)) #Seeds a random state
        random_index = np.random.permutation(data.shape[0]) #Assigns random values to each array value
        centroids = data[random_index[:self.n_clusters]] #Assigns centroids to random coordinates

        return centroids

    def get_random_centroids(self, data, labels): 
        """ Each centroid is the geometric mean of the data points that
            have that centroid's label. """
        centroids = np.zeros((self.n_clusters, data.shape[1])) #tuple(300,2)

        for i in range(self.n_clusters):
            centroids[i, :] = np.mean(data[labels == i, :], axis=0)

        return centroids

    def comp_distance(self, data, centroids):
        from numpy.linalg import norm
        """ For each point, it will compute the distance of each of the
            points to each of the centroids and return the square value. """
        distance = np.zeros((data.shape[0], self.n_clusters)) #10*[0.0.0.]

        for i in range(self.n_clusters):
            row_norm = norm(data - centroids[i, :], axis=1) #computes distance from data to centroids
            distance[:, i] = np.square(row_norm) #Calculate the square value of each point
            
        return distance

    def label_cluster(self, distance): #Get Label(self, data_set, centroids)
        """ For each element of the data set, chose the closest centroid. """
        return np.argmin(distance, axis=1) #Returns the index of the smallest value in the array

    def compute_ss_error(self, data, labels, centroids):
        from numpy.linalg import norm
        """ This function returns the lowest sum of squares error. """
        dist = np.zeros(data.shape[0])

        for i in range(self.n_clusters):
            #if the labels == i true, then normalize the value of the data's distance from the centroid
            dist[labels == i] = norm(data[labels == i] -
                                         centroids[i], axis=1)

        return np.sum(np.square(dist)) #Return the square of each point's distance

    def fit(self, data):
        """ This function fits centroids around the data until their converge.
            It does so by assigning labels on each point around the centroids.
            Finally, it assigns the sum of squares error. """
        self.centroids = self._initialize_centroids(data)

        """ Finish running until max number of iterations. """
        for i in range(self.max_iterations):
            old_centroids = self.centroids
            dist = self.comp_distance(data, old_centroids)
            """ Assign labels to each data point based on centroid distance. """
            self.labels = self.label_cluster(dist)
            self.centroids = self.get_random_centroids(data, self.labels)

            """ K-Means will finish running when centroids have converged. """
            if np.all(old_centroids == self.centroids):
                break

        self.error = self.compute_ss_error(data, self.labels, self.centroids)

    def predict(self, data):
        """ This function returns the centroids labels based on the distance. """
        distance = self.comp_distance(data, self.centroids)
        
        return self.label_cluster(distance)

from scipy.stats import norm
""" GMM Algorithm """
class GMM_Algorithm(object):
    from scipy.stats import norm
    """ The Gaussian Mixture Model (GMM) is a soft clustering and flexible model
    that allocates the likelihood of each data point belonging to each of the
    gaussians as well as an increased log liklihood that the closest data
    point to the closest cluster instead of a farther to another cluster or
    gaussian. """
    def __init__(self):
        np.random.seed(0)
        ran_num = np.linspace(-5, 5, num=100)
        self.c1 = ran_num * np.random.rand(len(ran_num)) + 10 #Creating cluster 1
        self.c2 = ran_num * np.random.rand(len(ran_num)) - 10 #Creating cluster 2
        self.c3 = ran_num * np.random.rand(len(ran_num))      #Creating cluster 3
        self.combined = np.stack((self.c1, self.c2, self.c3)).flatten()

        self.r = None
        self.pi = [1/3, 1/3, 1/3] #Creating pi divided by 3 clusters
        self.mu = [-5, 5, 3] #Creating an array mu to hold the mean of each cluster
        self.variance = [5, 3, 1] #Creating the variance for each mu

    def expectation(self):
        """ Expectation Maximization (EM) is a two step approach where the Expectation
            Step is used to calculate each data point xi to the probability r that, that
            data point in cluster c_i belongs to gaussian g_i. """
        r = np.zeros((len(self.combined), 3)) #Initializing array NxK (60, 3)

        """Initializing the random gaussians. 
        g1 = norm(loc=-5, scale=5)#loc = mu and scale = variance 
        g2 = norm(loc=1.5, scale=3)
        g3 = norm(loc=8, scale=1)"""

        for c, g, p in zip(range(3), #Range of c clusters, [gaussians], pi 
                           [norm(loc=self.mu[0], scale=self.variance[0]), #Gaussian 1
                            norm(loc=self.mu[1], scale=self.variance[1]), #Gaussian 2
                            norm(loc=self.mu[2], scale=self.variance[2])],#Gaussian 3
                           self.pi):                            
            r[:, c] = p * g.pdf(self.combined) #Each x_i has probability from cluster c_i belongs to g_i  

        for i in range(len(r)): #For each array pdf[x_i, x_i, x_i]
            r[i] = r[i]/(np.sum(self.pi) * np.sum(r, axis=1)[i]) #sum(pi) = 1 and sum(r) = 60 data points each data point has 3 x_i worth 1.0

        self.r = r
        
    def maximization(self):
        """ The Maximization Step then calculates the total weight m_c which is
            the fraction of points allocated to cluster c. """
        m_c = [] 
        for c in range(len(self.r[0])): #For each cluster c, in range 3 
            m = np.sum(self.r[:, c]) #The sum of all the points in cluster c_i 
            m_c.append(m) #Creating array containing the sum of each cluster c_i for each m_c 

        for i in range(len(m_c)): #For each mean in cluster c_i
            self.pi[i] = (m_c[i] / np.sum(m_c)) #For each cluster c_i, calculate the fraction of each point pi_c belonging to c
            
        #μ_c, for each data point x_i in cluster c_i (axis=0), multiply it by the probability of each r_i and divide it by
        self.mu = (np.sum(self.combined.reshape(len(self.combined), 1) * self.r, axis=0) / m_c)  

        var_c = []
        log_likelihood = []
        for c in range(len(self.r[0])): #For each cluster ci_i in probability r, Σc=mc(x−μc)T*(x−μc)
            var_c.append((1 / m_c[c]) *
                         np.dot( np.array(self.r[:, c].reshape(len(self.r), 1)) *
                                 (self.combined.reshape(len(self.combined), 1) - self.mu[c]).T, #Transpose of the Matrix
                                 (self.combined.reshape(len(self.combined), 1) - self.mu[c])))


        #log_likelihood.append(np.log(np.sum(multivariate_normal(self.mu.reshape(len(self.mu)), 1), var_c)))
        #Need to implement computation for log-likelihood
    
class KMeans_Main(object):
    """ This class will run the K-Means algorithm. """
    def __init__(self):
        self = self

    def run(self):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(6,6))
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('K-Means')

        k = KMeans_Algorithm(3)
        data = k.x
        sd = StandardScaler().fit_transform(data)
        k.fit(sd)
        predicted_centroids = k.predict(sd)
        centroids = k.centroids
        
        plt.scatter(sd[:, 0], sd[:, 1], c=predicted_centroids, s=70, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, alpha=0.5)
        #plt.show()

        iterations = 9 # math.ceil(11/3) or (12/3)
        fig, ax = plt.subplots(3, 4, figsize=(10,10))
        ax = np.ravel(ax)
        centers = []

        for i in range(iterations):
            k = KMeans_Algorithm(3, random_state=np.random.randint(0, 1000, size=1))
            k.fit(sd)
            centroids = k.centroids
            centers.append(centroids)
            ax[i].scatter(sd[:, 0], sd[:, 1], c=predicted_centroids, s=70, cmap='viridis')
            ax[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, alpha=0.5)
            ax[i].set_title(round(k.error, 5))

        plt.tight_layout()
        plt.show()

class GMM_Main(object):
    """ This class will run the K-Means algorithm. """
    def run(self):
        gmm = GMM_Algorithm()
        iterations = 10
        
        for i in range(iterations):
            gmm.expectation()

            """ Plotting the data points. """
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            #print(gmm.r[:, 0])

            for i in range(len(gmm.r)):
                ax.scatter(gmm.combined[i], 0, c=np.array([[gmm.r[i][0], gmm.r[i][1], gmm.r[i][2]]]), s=100)

            for g,c in zip([norm(loc=gmm.mu[0],scale=gmm.variance[0]).pdf(np.linspace(-20,20,num=60)),
                            norm(loc=gmm.mu[1],scale=gmm.variance[1]).pdf(np.linspace(-20,20,num=60)),
                            norm(loc=gmm.mu[2],scale=gmm.variance[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
                    ax.plot(np.linspace(-20,20,num=60),g,c=c)

            gmm.maximization()
        plt.show()
        

""" Main """
if __name__ == '__main__':
    kmain = KMeans_Main()
    #kmain.run()

    gmain = GMM_Main()
    gmain.run()




