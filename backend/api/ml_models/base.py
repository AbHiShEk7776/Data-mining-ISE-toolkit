class BaseClassifier:
    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y, metric_fn):
        y_pred = self.predict(X)
        return metric_fn(y, y_pred)
