import numpy as np
import statistics

class K_modes():

    def __init__(self, k:int, n_iters:int):
        self.k = k
        self.n_iters = n_iters
        np.random.seed(42)

    def fit(self, data):
        centroids = self.get_random_k_centroids(data)
        #centroids = np.asarray([['A','B','A','B','C'],
        #             ['C','C','C','B','A'],
        #             ['A','B','B','A','C']])
        print( f'First centroids: \n{centroids}')
        for _ in range(self.n_iters):
            distances = self.get_distance_to_centroids(data, centroids)
            labels = self.assign_to_cluster(distances)
            centroids = self.update_centroids(data, labels)
        return centroids

    def get_random_k_centroids(self, data):
        idx = np.random.choice(len(data), self.k, replace=False)
        centroids = data[idx, :]
        return centroids

    def get_distance_to_centroids(self, data, centroids):
        distances = list()
        for row in data:
            row_distances = list()
            for c in centroids:
                row_distances.append(self.compare_to_centroid(row, c))
            distances.append(row_distances)
        return distances
                
    def compare_to_centroid(self, row, centroid):
        diff = 0
        for c_r, c_c in zip(row,centroid):
            if c_r != c_c:
                diff += 1
        return diff

    def assign_to_cluster(self, distances):
        labels = list()
        for d in distances:
            labels.append(np.argmin(d))
        return labels

    def update_centroids(self, data, labels):
        out = list()
        for i in range(self.k):
            centroid = list()
            for row, label in zip(data,labels):
                if label == i:
                    centroid.append(row)
            centroid = np.asarray(centroid)
            new_col = list()
            for idx in range(len(centroid[0])):
                col = centroid[:,idx]
                new_col.append(statistics.mode(col))
            new_col = np.asarray(new_col)
            out.append(new_col)
        return np.asarray(out)


