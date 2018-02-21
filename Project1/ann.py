import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 17
        self.hiddenLayerSize = 17
        self.outputLayerSize = 26
        self.learn = 0.01

        # Layer arrays
        self.W1 = np.random.random_integers(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random_integers(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))
        self.S1 = None
        self.S2 = None
        self.Z1 = None
        self.Z2 = None
        self.SM = None

    def foward(self, X):
        # input to Layer1
        self.S1 = np.dot(X, self.W1)
        self.Z1 = self.sigmoid(self.S1)
        self.S2 = np.dot(self.Z1, self.W2)
        self.Z2 = self.sigmoid(self.S2) #self.Z2 = yhat
        self.SM = self.softmax(self.Z2)
        print("Z1", self.Z1.shape)
        print("W2", self.W2.shape)
        print("S2", self.S2.shape)
        return self.SM

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))  # 1 / (1 + e^-s)

    def sigmoid_prime(self, z):
        return np.exp(-z) / (1+np.exp(-z)**2)

    def softmax(self, z):
        soft = np.full(self.outputLayerSize, 0.10)
        max_index = np.argmax(z)
        soft[max_index] = 0.9
        return soft

    # Back popagating error (Delta)
    # DELTA[n] = -(y-y_hat) * sigmoid_prime(self.Z)
    ## delta[n] = (-Y - self.SM) *
    # where n refers to the layer starting with 0 for input
    def back_propagation(self, X, Y):
        # print(self.SM)
        # print("SM:",self.SM.shape,"Y:", Y.shape, "S2:", self.S2.shape)
        # print(np.subtract(Y,self.SM))
        delta2 = np.multiply(-(np.subtract(Y,self.SM)), self.sigmoid_prime(self.S2))
        print("D2",delta2)
        djdW2 = np.dot(self.Z1.T, delta2)

        delta = np.dot(delta2, self.W2.T)*self.sigmoid_prime(self.S1)
        djdW1 = np.dot(X.T, delta)
        self.W2 = self.W2 - self.learn*djdW2
        self.W1 = self.W1 - self.learn*djdW1


