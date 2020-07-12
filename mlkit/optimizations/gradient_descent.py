import numpy as np


class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def minimize_logistic_cost(self, X_train, cost, weights):
        gradient = np.dot(X_train.T, cost)
        average_gradient = gradient / X_train.shape[0]
        updated_weights = weights - self.learning_rate * average_gradient
        return updated_weights
