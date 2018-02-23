import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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
print("Learn Rate:", nn.learn)


def printStats(type, epochs, batchSize, wrong, mode, static):
    total_rows = batchSize * epochs
    percent_wrong = (wrong / total_rows) * 100

    if mode != 'simple':
        print()
        print(static, "MODE: ", type)
        print(static, "Epoch size:", epochs)
        print(static, "Batch size:", batchSize)
        print(static, "Number Wrong:", wrong)
        print(static, "Number possible:", total_rows)
        print(static, "Error percent:", percent_wrong,"%")
    print(static, "Accuracy percent:", (100 - percent_wrong),"%")


def validate(runNum):
    # Quick validation at the next block
    validation_errors = 0
    validation_range = 4000
    for i in range(16000, 16000 + validation_range):
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

    printStats('Validation', 1, validation_range, validation_errors, "simple", runNum)


# 1 complete run through the entire1 16000 training set is 1 epoch
def run_batches(xin, yin, batch_start, batch_size):
    total_wrong = 0
    while batch_start < len(xin):
        if batch_start + batch_size > len(xin):
            batch_size = len(xin) - batch_start

        total_wrong += run_ann(xin, yin, batch_start, batch_size)

        # Increment the starting position by the batch size
        batch_start += batch_size

    return total_wrong


def run_ann(xin, yin, batch_start, batch_size):
    total_wrong = 0
    x = xin[range(batch_start, batch_size), :]
    y = yin[range(batch_start, batch_size), :]

    yhat = nn.foward(x)  # 1 row at a time
    nn.back_propagation(x, y)  # Includes weight updates

    # Get the max array index for the 0-25 array (What letter)
    y_letter = np.argmax(y, axis=1)
    yhat_letter = np.argmax(yhat, axis=1)
    wrong_sum = y_letter - yhat_letter
    total_wrong = (wrong_sum != 0).sum()

    return total_wrong


def run_epochs(epochs, xin, yin, batch_start, batch_size):
    total_overall_errors = 0
    for e in range(epochs):
        total_overall_errors += run_batches(xin, yin, batch_start, batch_size)
        validate(str(e))

    printStats('Training', epochs, len(xin), total_overall_errors, "", "")

    return total_overall_errors


run_epochs(500, X, Y, 0, 16000)

# are the X values staying at 0?
print("X bias average", np.average(X[:, 0]))

actual_vs_predicted = np.full((26,26), 0)
validate("end")
plt.imshow(actual_vs_predicted, cmap='hot', interpolation='nearest')
plt.show()

print(actual_vs_predicted)
