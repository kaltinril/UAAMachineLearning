import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 17
        self.hiddenLayerSize = 17
        self.outputLayerSize = 26
        self.learn = 0.3

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
        self.Z2 = self.sigmoid(self.S2)
        self.yhat = self.softmax(self.S2)

        return self.yhat

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
        DEBUG = False
        print("Y shape:", Y.shape) if DEBUG else None
        self.yhat = self.yhat[np.newaxis]
        print("yhat Shape:",self.yhat.shape) if DEBUG else None
    
        # Output Layer Delta
        delta3 = (self.yhat - Y).T
        print("D3: Output Delta shape: ", delta3.shape) if DEBUG else None
        print("Z2SigPrim: ",self.sigmoid_prime(self.Z1).shape) if DEBUG else None
        
        # Hidden Layer Delta
        delta2 = np.dot(delta3, self.sigmoid_prime(self.Z1))
        print("D2: Hidden Delta shape: ", delta2.shape) if DEBUG else None
        
        # Input Layer Delta
        delta = np.dot(self.W2, delta2) * self.sigmoid_prime(self.S1).T
        print("D1: INput Delta shape: ", delta2.shape) if DEBUG else None

        print("W1 shape:",self.W1.shape) if DEBUG else None
        print("W2 shape:",self.W2.shape) if DEBUG else None
        
        
        # Weight updates
        djdW1 = np.dot(X, delta)
        print("DeltaWeight1", djdW1.shape) if DEBUG else None
        djdW2 = np.dot(self.Z1, delta2.T)
        print("DeltaWeight2", djdW2.shape) if DEBUG else None
        
        
        self.W2 = self.W2 + self.learn*djdW2
        self.W1 = self.W1 + self.learn*djdW1


