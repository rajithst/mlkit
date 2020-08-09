
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.size = None

    def minimize(self, gradient, weights):
        average_gradient = gradient / self.size
        updated_weights = weights - self.learning_rate * average_gradient
        return updated_weights
