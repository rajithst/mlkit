import numpy as np
from optimizations.gradient_descent import GradientDescentOptimizer

class LogisticRegression:
    def __init__(self, learning_rate,optimizer = GradientDescentOptimizer):
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def train(self, X_train, y_train,iterations=50):
        # train shape 200,3
        #weights shape 3,1
        self.weights = np.zeros((X_train.shape[1],1))
        cost_history = []
        for i in range(iterations):
            predictions = self.predict(X_train)
            prediction_cost = self.__logistic_cost(y_train, predictions)
            print(prediction_cost)
            cost_history.append(prediction_cost)

            grad_cost = y_train-predictions
            self.weights = self.optimizer(learning_rate=self.learning_rate).minimize_logistic_cost(X_train,grad_cost,self.weights)

    def predict(self, X):
        hypothesis = np.dot(self.weights, X)
        predicts = self.__sigmoid(hypothesis)
        return predicts

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))


    @staticmethod
    def __logistic_cost(true_labels, predictions):
        cost = -true_labels * np.log(predictions) - (1 - true_labels) * np.log(1 - predictions)
        cost = cost.sum() / len(true_labels)
        return cost