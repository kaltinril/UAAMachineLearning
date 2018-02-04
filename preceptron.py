import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.xlabel('Income')
plt.ylabel('Mortality')
plt.title('Linear Regression: Income vs. Infant Mortality')

data = pd.read_csv('./aurora_hist1.csv', header=None)
rows = data.shape [0]
cols = data.shape [1]
data = data.values
np.random.shuffle(data)  # Randomly adjust the rows
data = data[np.arange(0, rows), :]

# Remove the
X = data[:, (range(1, cols-1))]
W = [0.01]*len(X[0])  # Length of the columns in the first row
WZ = -0.001
TRN = 0
ACC = 0
os.system("pause")
Y = data[:, 0]
filenames = data[:, cols-1]
X = (X - X.min()) / (float(X.max()) - X.min())


# Normalize
# for i in range(0, len(X)):
#     print(X[0])
#     X[i] = (X[i] - min(X[i])) / (float(max(X[i])) - min(X[i]))

learn = 0.4
iterations = 100
batchStart = 0
batchSize = 5

# This is the Y Hat
# Y is the 1 or 0 for "Aurora = 1" and "Not = 0"
def perceptron_calculation(row_index):
    total = WZ
    x_values = X[row_index]

    for column in range(0, len(x_values) - 1):
        total += x_values[column] * W[column]

    print(total)
    os.system("pause")

    result = 0
    if total > 0:
        result = 1

    return result

# wb = wb - m (y^ – y) * x
def adjust_weight(row_index, y_hat):
    global WZ
    x_values = X[row_index]

    # Loop over all columns in the row
    for column in range(0, len(x_values) - 1):
        W[column] = W[column] - learn * (Y[row_index] - y_hat) * x_values[column]

    WZ = WZ - learn * (Y[row_index] - y_hat)


# TRAINING
# Loop over all rows
# Check if we are correct or not.
# If we are correct, continue
# if not, adjust weight
train_validate_split = int(rows * .8)
print(train_validate_split)
for row in range(0, train_validate_split):
    y_hat = perceptron_calculation(row)
    if Y[row] == y_hat:
        TRN += 1  # We got the correct prediction, increment by 1
    else:
        adjust_weight(row, y_hat)

    print(y_hat, TRN, row)

# Validation
for row in range(train_validate_split, rows):
    y_hat = perceptron_calculation(row)
    if Y[row] == y_hat:
        # Put the accuracy updates here
        ACC += 1  # We got the correct prediction, increment by 1

print("Training Correctness: " + str(TRN / train_validate_split))
print("Accuracy: " + str(ACC / (rows - train_validate_split)))

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
# plt.plot(X,Y, 'bo')
# plt.show()