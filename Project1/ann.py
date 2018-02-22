import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 17
        self.hiddenLayerSize = 16
        self.outputLayerSize = 26
        self.learn = 0.01

        # Layer arrays
        self.W1 = np.random.random_integers(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random_integers(-1, 1, (self.hiddenLayerSize+1, self.outputLayerSize))
        self.S1 = None
        self.S2 = None
        self.Z1 = None
        self.Z2 = None
        self.SM = None

    def foward(self, X):
        # input to Layer1
        self.S1 = np.dot(X, self.W1)
        self.Z1 = self.sigmoid(self.S1)

        # Add in a bias column
        self.Z1 = np.append(self.Z1, np.ones((self.Z1.shape[0], 1)), axis=1)

        self.S2 = np.dot(self.Z1, self.W2)
        #self.Z2 = self.sigmoid(self.S2) #self.Z2 = yhat
        self.Z2 = self.f_softmax(self.S2)

        #soft = self.f_softmax(self.S2)
        #print(soft)
        #print(np.sum(soft))
        #self.SM = self.softmax(self.Z2)

        #print("Z1", self.Z1.shape)
        #print("W2", self.W2.shape)
        #print("S2", self.S2.shape)
        return self.Z2

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))  # 1 / (1 + e^-s)

    def sigmoid_prime(self, z):
        return z*(1-z)  # np.exp(-z) / (1+np.exp(-z)**2)

    def softmax(self, z):
        soft = np.full(self.outputLayerSize, 0)
        max_index = np.argmax(z)
        soft[max_index] = 1
        return soft

    def f_softmax(self, X):
        Z = np.sum(np.exp(X), axis=1)
        Z = Z.reshape(Z.shape[0], 1)
        return np.exp(X) / Z

    # Back popagating error (Delta)
    # DELTA[n] = -(y-y_hat) * sigmoid_prime(self.Z)
    ## delta[n] = (-Y - self.SM) *
    # where n refers to the layer starting with 0 for input
    def back_propagation(self, X, Y):
        delta3 = (self.Z2 - Y)
        delta2 = np.multiply(delta3, self.sigmoid_prime(self.Z2))

        # Exclude bias column from delta calculation
        wnobias = self.W2[0:-1, :]
        delta = np.dot(wnobias, delta2.T) * self.sigmoid_prime(self.Z1)

        # Weight updates
        djdW1 = np.dot(X, delta.T)
        djdW2 = np.dot(self.Z1.T, delta2)

        self.W2 = self.W2 - self.learn*djdW2
        self.W1 = self.W1 - self.learn*djdW1

    def printShapes(self):

        print("W1", self.W1.shape)
        print("S1", self.S1.shape)
        print("Z1", self.Z1.shape)
        print("W2", self.W2.shape)
        print("S2", self.S2.shape)
        print("Z2", self.Z2.shape)
        print("SoftMax", self.SM.shape)

