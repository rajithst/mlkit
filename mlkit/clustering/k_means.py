import numpy as np

dataset = load_iris()
X = dataset.data
y = dataset.target


class KMeans:

    def __init__(self, n_clusters, n_init=10,iterations=1000, ):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.centroid_initials = n_init
        self.centroids = None
        self.cost_log = None

    def train(self, X):
        for init in self.centroid_initials:
            self.centroids = [X[np.random.randint(low=0, high=len(X))] for i in range(self.n_clusters)]
            for iter in self.iterations:
                assign_centroids = np.c_[np.arange(0, len(X)), np.zeros(len(X)) * np.nan]

                for i in range(len(X)):
                    distances = self.euclidean_distance(X[i], self.centroids)
                    assign_centroids[i, 1] = np.argmin(distances)

                for k in range(self.n_clusters):
                    cluster_data = X[assign_centroids[:, 1] == k]
                    new_cluster = np.mean(cluster_data, axis=0)
                    self.centroids[k] = new_cluster

    def predict(self):
        pass

    @staticmethod
    def euclidean_distance(X, centroids):
        return np.sqrt(np.sum(np.square(X - centroids), axis=1))
