import numpy as np

class ANN:
    def __init__(self):
        # Layer definitions
        self.inputLayerSize = 16
        self.hiddenLayerSize = 10
        self.hiddenLayerWidth = 1
        self.outputLayerSize = 26

        # Layer arrays
        self.W_in = np.random.random_integers(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W = np.random.random_integers(-1, 1, (self.hiddenLayerSize, self.hiddenLayerWidth))
        self.W_out = np.random.random_integers(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))
        self.S_in = None
        self.S = None
        self.S_out = None

    def foward_learning(self, X):
        # input to Layer1
        self.S_in = np.dot(X, self.W_in)
        self.Z_in = self.sigmoid(self.S_in)
        self.S = np.dot(self.Z_in, self.W[0])

        # Layer1 to Layer2 (N to N+!)

        # LayerFinal to output

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))  # 1 / (1 + e^-s)



# X = np.random.randn(1, 16)
# print(X)
# [[ 0.5493172   0.3859487   0.09700825 -0.70588039 -1.10151935  0.19557332
#    0.13441547 -0.10486223 -0.61647325  0.35607019  0.84703342  0.80315358
#   -1.40575479 -1.41993189  1.14724125 -0.72441052]]
# W = np.random.randn(16, 10)
# np.dot(X, W)
# array([[-2.67992058,  0.58073405,  0.63232844,  2.65453628, -2.0576537 ,
#         -1.29921952,  3.69145145,  0.00949092,  5.81982076, -0.41185868]])
