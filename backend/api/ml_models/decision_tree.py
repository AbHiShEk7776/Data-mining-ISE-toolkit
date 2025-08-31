import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    #  Impurity Measures 
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob_sq = (counts / counts.sum()) ** 2
        return 1 - prob_sq.sum()
    
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    # Information Gain
    def _information_gain(self, y, y_left, y_right, criterion):
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if criterion == 'gini':
            impurity_parent = self._gini(y)
            child_impurity = (n_left/n) * self._gini(y_left) + (n_right/n) * self._gini(y_right)
        else:  # entropy
            impurity_parent = self._entropy(y)
            child_impurity = (n_left/n) * self._entropy(y_left) + (n_right/n) * self._entropy(y_right)

        return impurity_parent - child_impurity

    #  Gain Ratio
    def _gain_ratio(self, y, y_left, y_right):
        ig = self._information_gain(y, y_left, y_right, criterion="entropy")  # GR only defined for entropy
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        split_info = 0
        for n_child in [n_left, n_right]:
            if n_child == 0:
                continue
            p = n_child / n
            split_info -= p * np.log2(p + 1e-9)

        return ig / (split_info + 1e-9)

    # Split Finder 
    def _best_split(self, X, y):
        best_feature, best_threshold, best_score = None, None, -1
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                if self.criterion == 'gini' or self.criterion == 'entropy':
                    score = self._information_gain(y, y_left, y_right, criterion=self.criterion)
                elif self.criterion == 'gain_ratio':
                    score = self._gain_ratio(y, y_left, y_right)
                else:
                    raise ValueError("Unknown criterion")

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_score

    #  Tree Builder
    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        if (depth >= self.max_depth or n_samples < self.min_samples_split or num_classes == 1):
            leaf_value = self._most_common_label(y)
            return {"leaf": True, "value": leaf_value}

        feature, threshold, score = self._best_split(X, y)

        if feature is None or score <= 0:
            leaf_value = self._most_common_label(y)
            return {"leaf": True, "value": leaf_value}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
