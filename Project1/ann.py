import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 17
        self.hiddenLayerSize = 17
        self.outputLayerSize = 26
        self.learn = 0.002

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
        self.yhat = self.f_softmax(self.S2) # maybe not pass this back??

        #print("S1 shape", self.S1.shape)
        
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
        print("yhat Shape:",self.yhat.shape) if DEBUG else None
        print("Z2:",self.Z2.shape) if DEBUG else None
    
        # Output Layer Delta
        # self.layers[-1].D = (yhat - labels).T
        delta3 = (self.Z2 - Y).T
        print("D3: Output Delta shape: ", delta3.shape) if DEBUG else None
        print("Z2SigPrim: ",self.sigmoid_prime(self.Z1).shape) if DEBUG else None
        
        # Hidden Layer Delta
        # i=1 (W2, Z2, D2)
        # self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * self.layers[i].Fp
        delta2 = self.W2.dot(delta3) * self.sigmoid_prime(self.Z1).T
        print("D2: Hidden Delta shape: ", delta2.shape) if DEBUG else None
        
        # Input Layer Delta
        delta = self.W1.dot(delta2) * self.sigmoid_prime(X).T
        print("D1: INput Delta shape: ", delta2.shape) if DEBUG else None

        print("W1 shape:",self.W1.shape) if DEBUG else None
        print("W2 shape:",self.W2.shape) if DEBUG else None
        
        # Weight updates
        wc2 = -self.learn * (delta3.dot(self.Z1)).T
        wc1 = -self.learn * (delta2.dot(X)).T
        
        print("DeltaWeight2", wc2.shape) if DEBUG else None
        print("DeltaWeight1", wc1.shape) if DEBUG else None
        
        
        self.W2 = self.W2 + wc2
        self.W1 = self.W1 + wc1


