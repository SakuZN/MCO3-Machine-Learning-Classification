import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearLrModel:
    def __init__(self, X, y, learning_rate=0.01, num_iterations=100, print_cost=False):
        self.X_train = X
        self.y_train = y
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.theta = np.zeros((self.n, 1))

    def train(self):
        for i in range(self.num_iterations):
            # calculate the hypothesis function
            h = sigmoid(np.dot(self.X_train, self.theta))

            # calculate the gradient
            grad = np.dot(self.X_train.T, (h - self.y_train)) / self.m

            # update the weights
            self.theta -= self.learning_rate * grad

            # print the cost every 100 iterations
            if self.print_cost and i % 100 == 0:
                cost = self.cost_function(h)
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X_test):
        # calculate the hypothesis function for the test data
        h = sigmoid(np.dot(X_test, self.theta))

        # classify the test data as 1 or 0 depending on the hypothesis value
        y_pred = np.where(h >= 0.5, 1, 0)

        return y_pred

    # calculate the binary cross entropy cost function
    def cost_function(self, h):
        # calculate the cost function
        cost = (-1 / self.m) * np.sum(self.y_train * np.log(h) + (1 - self.y_train) * np.log(1 - h))
        return cost


class SimpleLogicalRegression:
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model = LogisticRegression(random_state=16)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred


class DecisionTreeModel:
    def __init__(self, X, y, max_depth=5, criterion='gini', min_samples_split=2):
        self.X_train = X
        self.y_train = y
        self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.model = DecisionTreeClassifier(max_depth=self.max_depth,
                                            criterion=self.criterion, min_samples_split=self.min_samples_split)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def get_feature_importances(self):
        return self.model.feature_importances_
