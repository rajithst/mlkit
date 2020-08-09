import numpy as np
from optimizations.gradient_descent import GradientDescentOptimizer


class LogisticRegression:
    def __init__(self, learning_rate=0.01, optimizer=GradientDescentOptimizer):
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.learning_rate)
        self.cost_log = None

    def train(self, X_train, y_train, iterations=50, verbose=False):
        self.weights = np.zeros((X_train.shape[1], 1))
        self.optimizer.size = X_train.shape[0]
        cost_history = []
        for i in range(iterations):
            predictions = self.predict(X_train, True)
            prediction_cost = self.__logistic_cost(y_train, predictions)
            if verbose:
                print(prediction_cost)
            cost_history.append(prediction_cost)

            cost = predictions - y_train
            gradient = np.dot(X_train.T, cost) #derivative of cost function
            self.weights = self.optimizer.minimize(gradient, self.weights)
        self.cost_log = cost_history

    def predict(self, X, prob=False, threshold=0.5):
        hypothesis = np.dot(X, self.weights)
        predicts = self.__sigmoid(hypothesis)
        if prob is False:
            predicts = (predicts >= threshold).astype(int)
        return predicts

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __logistic_cost(true_labels, predictions):
        cost = -true_labels * np.log(predictions) - (1 - true_labels) * np.log(1 - predictions)
        return cost.mean()
