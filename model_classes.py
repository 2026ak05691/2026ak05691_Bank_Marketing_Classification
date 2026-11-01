import numpy as np

#############Logistic Regression##########################
class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):

            linear = np.dot(X, self.weights) + self.bias

            predictions = self.sigmoid(linear)

            dw = np.dot(X.T, predictions - y) / n_samples

            db = np.sum(predictions - y) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):

        linear = np.dot(X, self.weights) + self.bias

        return self.sigmoid(linear)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)

################ K-Nearest Neighbour implimentation ##############
class KNNScratch:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):

        probabilities = []

        for sample in X:

            distances = np.sqrt(
                np.sum((self.X_train - sample) ** 2, axis=1)
            )

            nearest = np.argsort(distances)[:self.k]

            nearest_labels = self.y_train[nearest]

            probability = np.mean(nearest_labels)

            probabilities.append(probability)

        return np.array(probabilities)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)

###################### Gaussian Navie Bayes ##################################
class GaussianNaiveBayesScratch:

    def fit(self, X, y):

        self.classes = np.unique(y)

        self.mean = {}
        self.variance = {}
        self.prior = {}

        for c in self.classes:

            X_class = X[y == c]

            self.mean[c] = np.mean(X_class, axis=0)

            self.variance[c] = np.var(X_class, axis=0) + 1e-9

            self.prior[c] = len(X_class) / len(X)

    def predict_proba(self, X):

        log_probabilities = []

        for c in self.classes:

            mean = self.mean[c]
            variance = self.variance[c]

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance)
                + ((X - mean) ** 2) / variance,
                axis=1
            )

            log_posterior = (
                np.log(self.prior[c]) + log_likelihood
            )

            log_probabilities.append(log_posterior)

        log_probabilities = np.array(log_probabilities).T

        # Convert log probabilities to probabilities
        log_probabilities -= np.max(
            log_probabilities, axis=1, keepdims=True
        )

        probabilities = np.exp(log_probabilities)

        probabilities /= np.sum(
            probabilities, axis=1, keepdims=True
        )

        return probabilities[:, 1]

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)

################### Decision Tree Classifier ###########################

class DecisionTreeScratch:

    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def gini(self, y):

        if len(y) == 0:
            return 0

        probabilities = np.bincount(
            y, minlength=2
        ) / len(y)

        return 1 - np.sum(probabilities ** 2)

    def best_split(self, X, y):

        best_gain = 0
        best_feature = None
        best_threshold = None

        parent_gini = self.gini(y)

        for feature in range(X.shape[1]):

            thresholds = np.unique(X[:, feature])

            if len(thresholds) > 20:
                thresholds = np.percentile(
                    X[:, feature],
                    np.linspace(0, 100, 21)
                )

            for threshold in thresholds:

                left = X[:, feature] <= threshold
                right = ~left

                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                weighted_gini = (
                    np.sum(left) / len(y) * self.gini(y[left])
                    + np.sum(right) / len(y) * self.gini(y[right])
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):

        counts = np.bincount(y, minlength=2)

        probabilities = counts / len(y)

        node = {
            "probability": probabilities[1],
            "prediction": int(np.argmax(counts))
        }

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return node

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return node

        left = X[:, feature] <= threshold
        right = ~left

        node["feature"] = feature
        node["threshold"] = threshold

        node["left"] = self.build_tree(
            X[left], y[left], depth + 1
        )

        node["right"] = self.build_tree(
            X[right], y[right], depth + 1
        )

        return node

    def fit(self, X, y):

        self.tree = self.build_tree(X, y)

    def predict_one(self, sample, node):

        if "feature" not in node:
            return node["probability"]

        if sample[node["feature"]] <= node["threshold"]:

            return self.predict_one(sample, node["left"])

        return self.predict_one(sample, node["right"])

    def predict_proba(self, X):

        return np.array([
            self.predict_one(sample, self.tree)
            for sample in X
        ])

    def predict(self, X):

        return (self.predict_proba(X) >= 0.5).astype(int)

############## Support Vector Machine ############################
class SVMScratch:

    def __init__(self, learning_rate=0.001,
                 iterations=1000, regularization=0.01):

        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_param = regularization

    def fit(self, X, y):

        n_samples, n_features = X.shape

        y_modified = np.where(y == 1, 1, -1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):

            for j in range(n_samples):

                condition = (
                    y_modified[j] *
                    (np.dot(X[j], self.weights) + self.bias)
                )

                if condition >= 1:

                    dw = 2 * self.lambda_param * self.weights

                    db = 0

                else:

                    dw = (
                        2 * self.lambda_param * self.weights
                        - y_modified[j] * X[j]
                    )

                    db = -y_modified[j]

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def decision_function(self, X):

        return np.dot(X, self.weights) + self.bias

    def predict_proba(self, X):

        scores = self.decision_function(X)

        scores = np.clip(scores, -500, 500)

        return 1 / (1 + np.exp(-scores))

    def predict(self, X):

        return (self.decision_function(X) >= 0).astype(int)
