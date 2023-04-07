from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE


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

    # plot coefficients per independent feature
    def plot_coefficients(self):
        plt.figure(figsize=(12, 12))

        coefs = self.model.coef_[0]
        plt.bar(self.X_train.columns, coefs)

        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.xlabel('Features')
        plt.ylabel('Coefficients')
        plt.title('Logistic Regression Coefficients per Feature')
        plt.show()

    def get_model(self):
        return self.model


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

    def graph_feature_importances(self):
        plt.figure(figsize=(12, 12))
        plt.bar(self.X_train.columns, self.model.feature_importances_)
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Decision Tree Feature Importances')
        plt.show()

    def generate_decisiontree_graph(self):
        dot_data = export_graphviz(self.model, out_file=None, feature_names=self.X_train.columns,
                                       class_names=['Employable', 'LessEmployable'], filled=True, rounded=True,
                                       special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Decision Tree", view=True)

    def get_model(self):
        return self.model


def get_cv_score(LR, DT, X, y, cv=5, scoring='accuracy'):
    lr_scores = cross_val_score(LR, X, y, cv=cv, scoring=scoring)
    dt_scores = cross_val_score(DT, X, y, cv=cv, scoring=scoring)
    return lr_scores, dt_scores

def visualize_cv_score(LR_score, DT_score, metric):
    cv_scores = [LR_score, DT_score]
    labels = ['Logistic Regression', 'Decision Tree']
    fig, ax = plt.subplots()
    ax.boxplot(cv_scores)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Scores')
    ax.set_title('Cross Validation Scores using metric: {0}'.format(metric))
    plt.show()