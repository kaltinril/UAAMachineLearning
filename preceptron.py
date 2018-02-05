import random


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

DEBUG = True
random.seed()
weights_filename = 'weights'

plt.xlabel('0-255')
plt.ylabel('count of hue')
plt.title('RGB histogram')

print("DEBUG: Loading histogram CSV") if DEBUG else None
data = pd.read_csv('./aurora_histogram.csv', header=None)

print("DEBUG: Processing data from CSV") if DEBUG else None
rows = data.shape [0]
cols = data.shape [1]
data = data.values





# Split the dataset into aurora and not aurora
# So we can correctly get 80% yes, and 80% no in the training set
aurora = []
not_aurora = []
for i in range(0, rows):
    if data[i][0] == 1:
        aurora.append(data[i])
    else:
        not_aurora.append((data[i]))

# Convert the arrays to numpy arrays
aurora = np.asarray(aurora)
not_aurora = np.asarray(not_aurora)

# Shuffle the arrays, so the order is random
np.random.shuffle(aurora)
np.random.shuffle(not_aurora)

# Pick out 80%
split_at = int(len(aurora) * .8)
AURORA_TRAIN = aurora[range(0, split_at)]
AURORA_VALIDATE = aurora[range(split_at, len(aurora))]

split_at = int(len(not_aurora) * .8)
NOT_AURORA_TRAIN = not_aurora[range(0, split_at)]
NOT_AURORA_VALIDATE = not_aurora[range(split_at, len(not_aurora))]

# combine the train sets and the validate sets
TRAIN_SET = np.concatenate((AURORA_TRAIN, NOT_AURORA_TRAIN), axis=0)
VALIDATE_SET = np.concatenate((AURORA_VALIDATE, NOT_AURORA_VALIDATE), axis=0)

# Shuffle the train and validation sets
np.random.shuffle(TRAIN_SET)
np.random.shuffle(VALIDATE_SET)

# Shove the arrays ontop since our code picks out the 80%
data = np.concatenate((TRAIN_SET, VALIDATE_SET), axis=0)

print("DEBUG: Randomizing data order from CSV") if DEBUG else None
#np.random.shuffle(data)  # Randomly adjust the rows

print("DEBUG: Creating DATA multi-dementional-array") if DEBUG else None
#data = data[np.arange(0, rows), :]

# Remove the
X = data[:, (range(1, cols-1))]
Y = data[:, 0]
W = [0]*len(X[0])  # Length of the columns in the first row
for i in range(0, len(W)):
    W[i] = random.randint(-1, 1)

W = np.genfromtxt(weights_filename + ".csv", delimiter=',')

WZ = random.randint(-1, 1)
TRN = 0
ACC = 0
learn = 0.01
iterations = 1000
batchStart = 0
batchSize = 5

filenames = data[:, cols-1]

# plt.plot(X[0][range(0, 255)], 'b')
# plt.plot(X[0][range(256, 511)], 'g')
# plt.plot(X[0][range(512, 768)], 'r')
# plt.show()

X = (X - X.min()) / (float(X.max()) - X.min())

# plt.plot(X[0][range(0, 255)], 'b')
# plt.plot(X[0][range(256, 511)], 'g')
# plt.plot(X[0][range(512, 768)], 'r')
# plt.show()


# This is the Y Hat
# Y is the 1 or 0 for "Aurora = 1" and "Not = 0"
def perceptron_calculation(histograms, row_index):
    x_values = histograms[row_index]

    # total = W0 + sum(X[i]*W[i])
    result_array = np.multiply(x_values, W)
    total = np.sum(result_array)
    total = total + WZ

    result = 1
    if total < 0.5:
        result = 0

    return result


# wb = wb - m (y^ – y) * x
def adjust_weight(histograms, hist_y, row_index, y_hat):
    global WZ
    x_values = histograms[row_index]

    # Hoping the pre-computed values will help with speed
    error_value = y_hat - hist_y[row_index]
    learn_error = learn * error_value

    # Loop over all columns in the row
    for column in range(0, len(x_values)):
        W[column] = W[column] - (learn_error * x_values[column])

    WZ = WZ - learn_error


# TRAINING
# Loop over all rows
# Check if we are correct or not.
# If we are correct, continue
# if not, adjust weight

train_validate_split = int(rows * .8)
for its in range(0, iterations):
    for row in range(0, train_validate_split):
        y_hat = perceptron_calculation(X, row)
        if Y[row] == y_hat:
            TRN += 1  # We got the correct prediction, increment by 1
        else:
            adjust_weight(X, Y, row, y_hat)

    print("TRN:", TRN, "Iter", its)

# Validation
for row in range(train_validate_split, rows):
    y_hat = perceptron_calculation(X, row)
    if Y[row] == y_hat:
        # Put the accuracy updates here
        ACC += 1  # We got the correct prediction, increment by 1

TRN_PER = (TRN / iterations) / train_validate_split
ACC_PER = ACC / (rows - train_validate_split)
print("Training Correctness: " + str(TRN_PER))
print("Accuracy: " + str(ACC_PER))

# for i in range(iterations):
#     bStart = batchStart
#     bSize = batchSize
#     while(bStart < len(X)):
#         if (bStart + bSize > len(X)):
#             bSize = len(X) - bStart
#         s1, s2 = summation(y,X,Y, bStart, bSize)
#         bStart += bSize
#         b1 = b1 - learn * s2
#         b0 = b0 - learn * s1
#         plt.plot(X, b1 * X + b0)
#         print("b0: " + str(b0) + " b1: " + str(b1) + " error: " + str(s2))
# for i in range(0, len(X)):
#    plt.plot(X[i][range(0, 255)], 'b')
#    plt.plot(X[i][range(256, 511)], 'g')
#    plt.plot(X[i][range(512, 768)], 'r')
# plt.plot(range(0,255), X[:, range(0,255)], 'bo')
# plt.plot(range(0,255), X[:, range(256, 511)], 'go')
# plt.plot(range(0,255), X[:, range(512, 768)], 'ro')
W = np.asarray(W)
plt.plot(W[range(0,255)], 'b')
plt.plot(W[range(256, 511)], 'g')
plt.plot(W[range(512, 768)], 'r')
plt.show()

plt.plot(W[range(0,255)], 'bo')
plt.plot(W[range(256, 511)], 'go')
plt.plot(W[range(512, 768)], 'ro')
plt.show()

ACC_STR = str(int(ACC_PER * 100))

np.savetxt(weights_filename + ACC_STR + ".csv", W, delimiter=',')
output_file = open('weight_zero' + ACC_STR + '.csv', 'w')
output_file.write(WZ)
output_file.close()
print(WZ)
