import numpy as np
import pandas as pd
import ann
#np.set_printoptions(threshold=np.nan)

data = pd.read_csv('./letters.csv', header=None)

rows = data.shape[0]
cols = data.shape[1]
data = data.values
numberWrong = 0

X = np.array(data[:, (range(1, cols))], dtype = float)
X = np.insert(X,0,1,axis=1) # adding the bias
Y_array = data[:, 0] # Snag the first column corresponding to the letter
# Y = np.zeros([Y_array.shape[0], 26], dtype = int)
Y = np.full([Y_array.shape[0], 26], 0.1)
for i in range(Y_array.shape[0]):
    Y[i,Y_array[i] - 1] = 0.9

# Normalizing each feature
for i in range(1, 17):
    X[:,i] = (X[:,i] - X[:,i].min()) / (float(X[:,i].max()) - X[:,i].min())

nn = ann.ANN()
batch_size = 500
for i in range(0, batch_size):
    # Fix the issue with Numpy array being (17,) instead of (1,17)
    x = X[i].reshape(X[i].shape[0], 1).T
    # print("X Shape",x.shape)
    y = Y[i].reshape(Y[i].shape[0], 1).T
    yhat = nn.foward(x) # 1 row at a time
    y_letter = np.argmax(y)
    yhat_letter = np.argmax(yhat)
    print(y_letter, yhat_letter)
    if y_letter != yhat_letter:
        nn.back_propagation(x, y)
        numberWrong += 1

print(yhat)
print("Batch size:", batch_size)
print("Number Wrong:",numberWrong)
percent_wrong = (numberWrong / batch_size) * 100
print("Error percent:",percent_wrong,"%")
print("Accuracy percent:", (100 - percent_wrong),"%")