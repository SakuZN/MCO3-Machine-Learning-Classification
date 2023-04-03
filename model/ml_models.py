import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures

class SimpleLogicalRegression:
    def __init__(self, X, y, random_state=16, C=1.0,
                 penalty='l2', solver='lbfgs', max_iter=100):
        self.X_train = X
        self.y_train = y
        self.model = LogisticRegression(random_state=random_state, penalty=penalty,
                                        C=C, solver=solver, max_iter=max_iter)

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
