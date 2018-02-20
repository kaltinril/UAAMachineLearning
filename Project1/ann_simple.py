import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 16
        self.hiddenLayerSize = 20
        self.outputLayerSize = 26

        # Layer arrays
        self.W1 = np.random.random_integers(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random_integers(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))
        self.S1 = None
        self.S2 = None
        self.Z1 = None
        self.Z2 = None
        self.SM = None

    def foward_learning(self, X):
        # input to Layer1
        self.S1 = np.dot(X, self.W1)
        self.Z1 = self.sigmoid(self.S1)

        # LayerFinal to output
        self.S2 = np.dot(self.Z1, self.W2)
        self.Z2 = self.sigmoid(self.S2)

        print("Z2", self.Z2)
        print("Z1", self.Z1)
        self.SM = self.softmax(self.Z2)
        return self.SM

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))  # 1 / (1 + e^-s)

    # Derivitive of the sigmoid function
    # sigmoid(s) = (u'v -uv') / v^2
    #   where u = 1, v = 1 + e^(-s)
    #   (0 * v - 1 * -e^(-s)) / (1 + e^-s)^2
    def sigmoid_prime(self, z):
        return np.exp(-z) / (1+np.exp(-z)**2)

    def softmax(self, z):
        soft = np.full((self.outputLayerSize, 1), 0.10)
        max_index = np.argmax(z)
        soft[max_index] = 0.9
        return soft

    # Back popagating error (Delta)
    # DELTA[n] = -(y-y_hat) * sigmoid_prime(self.Z)
    ## delta[n] = (-Y - self.SM) *
    # where n refers to the layer starting with 0 for input
    def back_propagation(self, X, y):
        delta2 = np.multiply(-(Y-self.SM))
