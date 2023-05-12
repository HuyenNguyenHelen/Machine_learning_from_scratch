import numpy as np
import scipy
from scipy import stats 
from sklearn.neighbors import KNeighborsClassifier

class KNN():
    """
    1. calculate distance between the new datapoint with each instance in the Train data.
        - for x in X_train:
            euclidean_dist = sqrt(sum((newpoint-x)**2))
        - store in sorted dictionary or tuples
    2. select the k top nearest by taking the top k min euclidean_dist
    3. make prediction based on mode of nearest neighbors' class.
    """
    def __init__(self, k) -> None:
        self.k = k

    # def fit(self, X_train, y_train):
    def fit(self, X_train, y_train, newpoint):
        self.X_train = X_train
        self.y_train = y_train
        self.newpoint = newpoint

        distances = {}
        for i, x in enumerate(X_train):
            dist = self.get_euclidean(x, self.newpoint)
            distances[i] = dist
        sorted_dist = {k:v for k, v in sorted(distances.items(), key = lambda item: item[1], reverse = False )}
        # print(sorted_dist)
        return sorted_dist
    
    def predict(self):
        sorted_dist = self.fit(self.X_train, self.y_train, self.newpoint)
        nearest_idx = list(sorted_dist.keys())[:self.k]
        nearest_labels = self.y_train[nearest_idx]
        # print(nearest_labels)
        pred_label = stats.mode(nearest_labels)
        # print('pred_label', pred_label.mode[0])
        return pred_label.mode[0]


    def get_euclidean(self, point1, point2):
        dist = np.sqrt(sum((point1 - point2)**2))
        return dist
    
    def get_accuracy(self, true_y, pred_y):
        if len(true_y)!=len(pred_y):
            raise ValueError(f'Length of two inputs must be the same. Len(true_y) = {len(true_y)}, len(pred_y) = {len(pred_y)}')
        else:
            n_correct = 0
            for i in range(len(true_y)):
                if true_y[i] == pred_y[i]:
                    n_correct+=1
                else: continue
            accuracy = n_correct/len(true_y)
        return accuracy



np.random.seed(42)
X = np.random.random_sample((100, 5))
y = np.random.choice(3, 100)
print(X.shape, y.shape)
knn = KNN(k = 3)
pred_y = []
for x in X:
    neighbors = knn.fit(X, y, x)
    pred_y.append(knn.predict())

print(y)
print(pred_y)
print('accuracy = ', knn.get_accuracy(y, pred_y))

# compare with sklearn
sk_knn = KNeighborsClassifier(n_neighbors=3)
sk_knn.fit(X, y)
sk_knn_pred = sk_knn.predict(X)
print('accuracy = ', knn.get_accuracy(y, sk_knn_pred))
