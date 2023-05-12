import numpy as np
import pandas as pd


class Kmeans(): 
    """
    1. initialize by a random data point from the dataset as seeding centroid. k is N random datapoints
    2. assign each datapoint to the nearest cluster based on the min distance between the datapoint and the centroid.
            - for all datapoints (exluding the choosen centroid datapoint) and centroid
                + calculate the distance between the datapoint and the centroid.
                + based on the min to assign cluster for the datapoint
    3. recalculate again the centroid for each cluster by taking the avg of all datapoints in each cluster
    4. repeat step 2 and 3 until there is no more cluster assigning change or no more changes in centroid position. 
    """
    def __init__(self, data, k, max_iter) -> None:
        self.data = data
        self.k = k
        self.max_iter = max_iter
        # 1.initialize by a random data point from the dataset as seeding centroid. k is N random datapoints
        rand_idx = np.random.choice(len(self.data), k)
        self.centroids = self.data[rand_idx, :]
        print('init centroids', self.centroids)

    def get_euclidean(self, data_point_1, data_point_2):
        return np.sqrt(sum((np.array(data_point_1) - np.array(data_point_2))**2))
    
    def fit(self):
        # 2. assign each datapoint to the nearest cluster based on the min distance between the datapoint and the centroid.
        # for all datapoints (exluding the choosen centroid datapoint) and centroid
        # calculate the distance between the datapoint and the centroid.
        assigned_cluster = {i: [] for i in range (self.k)}
        for i, x in enumerate(self.data):
            distances = self.get_euclidean(x, self.centroids)
            # based on the min to assign cluster for the datapoint
            selected_centroid_idx = np.argmin(distances)
            assigned_cluster[selected_centroid_idx].append(i)

        # 3. determine again the centroid for each cluster by taking the avg of all datapoints in each cluster
        pre_centroids = None
        iter = 0
        while iter < self.max_iter and np.not_equal(pre_centroids, self.centroids).any():
            print(f'------------- iteration {iter} -------------')
            pre_centroids = self.centroids
            self.centroids = [np.mean(self.data[value, :], axis = 0) for key, value in assigned_cluster.items()]
            print('new centroid', self.centroids)
            updated_clusters = {i: [] for i in range (self.k)}
            for i, x in enumerate(self.data):
                distances = self.get_euclidean(x, self.centroids)
                # based on the min to assign cluster for the datapoint
                selected_centroid_idx = np.argmin(distances)
                updated_clusters[selected_centroid_idx].append(i)

            assigned_cluster = updated_clusters
            iter+=1
        return assigned_cluster
    def get_cluster(self, datapoints):
        
        if datapoints.shape[0]>1:
            clusters = {}
            for i, x in enumerate(datapoints):
                distances = self.get_euclidean(x, self.centroids)
                selected_centroid_idx = np.argmin(distances)
                clusters[i] = selected_centroid_idx
            return clusters
        else:
            distances = self.get_euclidean(datapoints, self.centroids)
            selected_centroid_idx = np.argmin(distances)
            return selected_centroid_idx





np.random.seed(42)
data = np.random.random_sample((100, 3))
kmeans = Kmeans(data,3, 200)
train_clusters = kmeans.fit()
print(kmeans.get_cluster(np.array([[0.4, 1, 0.9]])))



