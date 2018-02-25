import numpy as np

class ANN:
    def __init__(self, input_size=16, hidden_size=21, output_size=26, learn=0.01):
        # Layer definitions
        self.inputLayerSize = input_size
        self.hiddenLayerSize = hidden_size
        self.outputLayerSize = output_size
        self.learn = learn
        # 1 = errors
        # 0.1 = slow learning, peaks around 35% accuracy
        # 0.05 = 50% around 50, then just jumps between 49-51 for remainder of 500 batch
        # 0.01 = fast, got to 70% in 100 runs, peaked around 71%
        # 0.005 71% 275 epochs at 100 batch size
        # 0.001 slower than 0.005, peaks around 63% around 350 epochs
        # 0.0005 again better than the above and below numbers, got to 63% around 350
        # 0.0001 super slow 53% around 500
        # Need to test with other values.

        # Layer arrays
        self.W1 = np.random.random_integers(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random_integers(-1, 1, (self.hiddenLayerSize+1, self.outputLayerSize))
        self.S1 = None
        self.S2 = None
        self.Z1 = None
        self.Z2 = None

    def foward(self, X):
        # input to Layer1
        self.S1 = np.dot(X, self.W1)
        self.Z1 = self.sigmoid(self.S1)

        # Add in a bias column
        self.Z1 = np.append(self.Z1, np.ones((self.Z1.shape[0], 1)), axis=1)

        self.S2 = np.dot(self.Z1, self.W2)
        self.Z2 = self.sigmoid(self.S2)
        self.yhat = self.f_softmax(self.Z2)
        
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
        delta3 = (self.Z2 - Y).T
        print("D3: Output Delta shape: ", delta3.shape) if DEBUG else None
        print("Z2SigPrim: ",self.sigmoid_prime(self.Z1).shape) if DEBUG else None

        # Exclude bias column from delta calculation
        w2nobias = self.W2[0:-1, :]
        z1nobias = self.Z1[:, 0:-1]

        # Hidden Layer Delta
        delta2 = w2nobias.dot(delta3) * self.sigmoid_prime(z1nobias).T
        print("D2: Hidden Delta shape: ", delta2.shape) if DEBUG else None

        print("W1 shape:",self.W1.shape) if DEBUG else None
        print("W2 shape:",self.W2.shape) if DEBUG else None
        
        # Weight updates
        wc2 = -self.learn * (delta3.dot(self.Z1)).T
        wc1 = -self.learn * (delta2.dot(X)).T
        
        print("DeltaWeight2", wc2.shape) if DEBUG else None
        print("DeltaWeight1", wc1.shape) if DEBUG else None

        self.W2 = self.W2 + wc2
        self.W1 = self.W1 + wc1
