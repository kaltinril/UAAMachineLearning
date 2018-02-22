import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Create a 26x26 array for our heatmap
actual_vs_predicted = np.full((26,26), 0)

nn = ann.ANN()
epocs = 50
iterations = 100
batch_size = 500
for h in range(epocs):
    for j in range(0, iterations):
        for i in range(0, batch_size):
            # Fix the issue with Numpy array being (17,) instead of (1,17)
            x = X[i].reshape(X[i].shape[0], 1).T
            y = Y[i].reshape(Y[i].shape[0], 1).T

            yhat = nn.foward(x) # 1 row at a time

            # Get the max array index for the 0-25 array (What letter)
            y_letter = np.argmax(y)
            yhat_letter = np.argmax(yhat)

            # Store the values so we can create a 2D heat map
            # actual_vs_predicted[y_letter, yhat_letter] += 1

            nn.back_propagation(x, y)

            # If our prediction was wrong, back propogate
            if y_letter != yhat_letter:
                numberWrong += 1

# QUick validation at the next block
validation_errors = 0
for i in range(batch_size, batch_size * 2):
    # Fix the issue with Numpy array being (17,) instead of (1,17)
    x = X[i].reshape(X[i].shape[0], 1).T
    y = Y[i].reshape(Y[i].shape[0], 1).T

    yhat = nn.foward(x)  # 1 row at a time

    # Get the max array index for the 0-25 array (What letter)
    y_letter = np.argmax(y)
    yhat_letter = np.argmax(yhat)

    # Store the values so we can create a 2D heat map
    actual_vs_predicted[y_letter, yhat_letter] += 1

    # If we were wrong, calulcate that
    if y_letter != yhat_letter:
        validation_errors += 1


#print(actual_vs_predicted)

def printStats(type, epochs, iters, batchSize, wrong):
    total_rows = iters * batchSize * epochs
    percent_wrong = (wrong / total_rows) * 100
    print("MODE: ", type)
    print("Epoch size:", epochs)
    print("Iterations:", iters)
    print("Batch size:", batchSize)
    print("Number Wrong:",wrong)
    print("Number possible:", total_rows)
    print("Error percent:",percent_wrong,"%")
    print("Accuracy percent:", (100 - percent_wrong),"%")
    print()


printStats('Training', epocs, iterations, batch_size, numberWrong)
printStats('Validation', 1, 1, batch_size, validation_errors)

plt.imshow(actual_vs_predicted, cmap='hot', interpolation='nearest')
plt.show()

print(actual_vs_predicted)
