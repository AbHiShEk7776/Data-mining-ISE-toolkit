import numpy as np

class RuleBasedClassifier:
    def __init__(self, rules=None):
        """
        rules: list of tuples (rule_function, class_label)
        rule_function: function(X_row) -> True/False
        class_label: label to assign if rule matches
        """
        self.rules = rules if rules is not None else []
        self.default_class = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        For simplicity, set default class as majority class.
        User can still add custom rules.
        """
        classes, counts = np.unique(y, return_counts=True)
        self.default_class = classes[np.argmax(counts)]

    def add_rule(self, rule_function, class_label):
        self.rules.append((rule_function, class_label))

    def predict(self, X: np.ndarray):
        preds = []
        for row in X:
            matched = False
            for rule_func, class_label in self.rules:
                if rule_func(row):
                    preds.append(class_label)
                    matched = True
                    break
            if not matched:
                preds.append(self.default_class)
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
