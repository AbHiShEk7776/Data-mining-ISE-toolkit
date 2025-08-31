import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.mean = {}
        self.variance = {}
        self.classes = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes, class_counts = np.unique(y, return_counts=True)
    
        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.variance[cls] = np.var(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / n_samples
                
    def _gaussian_prob(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + 1e-9))
        
        return (1 / np.sqrt(2 * np.pi * var + 1e-9)) * exponent
    
    def _posterior(self, x):
        posteriors = {}
        
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = np.sum(np.log(self._gaussian_prob(x, self.mean[cls], self.variance[cls])))
            posterior = prior + likelihood
            posteriors[cls] = posterior
            
        return posteriors
    
    def predict(self, X):
        return np.array([max(self._posterior(x), key=self._posterior(x).get) for x in X])
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)