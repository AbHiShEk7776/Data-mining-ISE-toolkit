import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))
        
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common = values[np.argmax(counts)]
            predictions.append(most_common)
        return np.array(predictions)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)