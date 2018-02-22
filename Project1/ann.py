import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 17
        self.hiddenLayerSize = 17
        self.outputLayerSize = 26
        self.learn = 0.1

        # Layer arrays
        self.W1 = np.random.random_integers(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random_integers(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))
        self.S1 = None
        self.S2 = None
        self.Z1 = None
        self.Z2 = None

    def foward(self, X):
        # input to Layer1
        self.S1 = np.dot(X, self.W1)
        self.Z1 = self.sigmoid(self.S1)
        self.S2 = np.dot(self.Z1, self.W2)
        self.Z2 = self.softmax(self.S2)

        return self.Z2

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))  # 1 / (1 + e^-s)

    def sigmoid_prime(self, z):
        return z*(1-z)  # np.exp(-z) / (1+np.exp(-z)**2)

    def f_softmax(self, X):
        Z = np.sum(np.exp(X), axis=1)
        Z = Z.reshape(Z.shape[0], 1)
        return np.exp(X) / Z

    def softmax(self, z):
        soft = np.full(self.outputLayerSize, 0.10)
        max_index = np.argmax(z)
        soft[max_index] = 0.9
        return soft

    # Back popagating error (Delta)
    def back_propagation(self, X, Y):
        delta3 = self.Z2 - Y
        delta2 = np.multiply(delta3, self.sigmoid_prime(self.Z2))
        delta = np.dot(delta2, self.W2.T) * self.sigmoid_prime(self.Z1)

        # Weight updates
        djdW1 = np.dot(X.T, delta)
        djdW2 = np.dot(self.Z1.T, delta2)

        self.W2 = self.W2 - self.learn*djdW2
        self.W1 = self.W1 - self.learn*djdW1


