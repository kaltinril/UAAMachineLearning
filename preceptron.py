import random


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEBUG = True
random.seed()
weights_filename = 'weights.csv'
weights_zero_filename = 'weight_zero.csv'

print("DEBUG: Loading histogram CSV") if DEBUG else None
data = pd.read_csv('./aurora_histogram.csv', header=None)

print("DEBUG: Processing data from CSV") if DEBUG else None
rows = data.shape[0]
cols = data.shape[1]
data = data.values

# Split the dataset into aurora and not aurora
# So we can correctly get 80% yes, and 80% no in the training set
print("DEBUG: Separate out the AURORA and NOT AURORA records") if DEBUG else None
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
print("DEBUG: Randomize the order of the AURORA and NOT AURORA rows") if DEBUG else None
np.random.shuffle(aurora)
np.random.shuffle(not_aurora)

# Pick out 80%
print("DEBUG: Pick out 80% 20% for aurora and not aurora for train and validation") if DEBUG else None
split_at = int(len(aurora) * .8)
AURORA_TRAIN = aurora[range(0, split_at)]
AURORA_VALIDATE = aurora[range(split_at, len(aurora))]

split_at = int(len(not_aurora) * .8)
NOT_AURORA_TRAIN = not_aurora[range(0, split_at)]
NOT_AURORA_VALIDATE = not_aurora[range(split_at, len(not_aurora))]

# combine the train sets and the validate sets
print("DEBUG: Combine the train sets and the validate sets") if DEBUG else None
TRAIN_SET = np.concatenate((AURORA_TRAIN, NOT_AURORA_TRAIN), axis=0)
VALIDATE_SET = np.concatenate((AURORA_VALIDATE, NOT_AURORA_VALIDATE), axis=0)

# Shuffle the train and validation sets
print("DEBUG: Randomizing separated train and validation sets") if DEBUG else None
np.random.shuffle(TRAIN_SET)
np.random.shuffle(VALIDATE_SET)

# Shove the arrays ontop since the code picks out the 80%
print("DEBUG: Combining the train and validation set to 1 array") if DEBUG else None
data = np.concatenate((TRAIN_SET, VALIDATE_SET), axis=0)

# Setup the X, W, Y, and filename arrays
print("DEBUG: Setting up the X, W, Y, and Filename arrays") if DEBUG else None
X = data[:, (range(1, cols-1))]
Y = data[:, 0]
filenames = data[:, cols-1]
WZ = random.randint(-1, 1)
W = [0]*len(X[0])  # Length of the columns in the first row
for i in range(0, len(W)):
    W[i] = random.randint(-1, 1)

# Normalize the data
print("DEBUG: Normalizing the X values") if DEBUG else None
X = (X - X.min()) / (float(X.max()) - X.min())

# Parameters
learn = 0.01
iterations = 100  # This is the "number of batches" essentially now
batch_size = 100

# Arrays and variables for graphing or analysis
TRN = 0
ACC = 0
trn_batch_error = []
val_batch_error = []
tbe_avg = []
vbe_avg = []
failed_filenames = []

# Load weights from file instead
# W = np.genfromtxt(weights_filename, delimiter=',')
# handle = open(weights_zero_filename, 'r')
# WZ = float(handle.readline())


# This is the Y Hat
# Y is the 1 or 0 for "Aurora = 1" and "Not = 0"
def perceptron_calculation(histograms, row_index):
    x_values = histograms[row_index]

    # Calculate the Perceptron value
    # Below is doing the following: total = W0 + sum(X[i]*W[i])
    result_array = np.multiply(x_values, W)
    total = np.sum(result_array)
    total = total + WZ

    result = 0
    if total > 0:
        result = 1

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


train_validate_split = int(rows * .8)
tbavgcnt = 0
vbavgcnt = 0
for its in range(0, iterations):
    batch_training_error_count = 0
    batch_validate_error_count = 0
    tavgsum = 0
    vavgsum = 0

    # Training batch
    batch_value = 0
    while batch_value < batch_size:
        # Pick random value for next row to use in training
        picked_training_row = random.randint(0, train_validate_split - 1)

        y_hat = perceptron_calculation(X, picked_training_row)
        if Y[picked_training_row] == y_hat:
            TRN += 1  # We got the correct prediction, increment by 1
        else:
            adjust_weight(X, Y, picked_training_row, y_hat)
            batch_training_error_count += 1

        # Increment the batch
        batch_value += 1

    tavgsum += (batch_training_error_count / batch_size)

    # Store the average error for all predictions
    trn_batch_error.append(batch_training_error_count / batch_size)

    # Validation Batch
    batch_value = 0
    while batch_value < batch_size:
        # Pick a random value for the next row to use in the validation
        picked_validation_row = random.randint(train_validate_split, rows - 1)

        y_hat = perceptron_calculation(X, picked_validation_row)
        if Y[picked_validation_row] == y_hat:
            # Put the accuracy updates here
            ACC += 1  # We got the correct prediction, increment by 1
        else:
            batch_validate_error_count += 1
            if its == (iterations - 1):
                failed_filenames.append(filenames[picked_validation_row])

        # Increment the batch
        batch_value += 1

    vavgsum += batch_validate_error_count / batch_size

    # Store the average error for all predictions
    val_batch_error.append(batch_validate_error_count / batch_size)

    # print("Iteration: " + str(its),
    #       "Train err%:", str(batch_training_error_count / batch_size),
    #      "Valid err%:", str(batch_validate_error_count / batch_size))

    avg_across = iterations / 10
    tbavgcnt = tbavgcnt % avg_across
    vbavgcnt = vbavgcnt % avg_across
    tbavgcnt += 1
    vbavgcnt += 1

    if tbavgcnt == avg_across:
        tbe_avg.append(tavgsum / avg_across)
        vbe_avg.append(vavgsum / avg_across)

TRN_PER = TRN / (iterations * batch_size)
ACC_PER = ACC / (iterations * batch_size)
print("Training Error %: " + str(1-TRN_PER))
print("Validate Error %: " + str(1-ACC_PER))


W = np.asarray(W)
# plt.plot(W[range(0,255)], 'b')
# plt.plot(W[range(256, 511)], 'g')
# plt.plot(W[range(512, 768)], 'r')
# plt.show()
#
# plt.plot(W[range(0,255)], 'bo')
# plt.plot(W[range(256, 511)], 'go')
# plt.plot(W[range(512, 768)], 'ro')
# plt.show()

np.savetxt(weights_filename, W, delimiter=',')
output_file = open(weights_zero_filename, 'w')
output_file.write(str(WZ))
output_file.close()
print(WZ)

print(len(failed_filenames))
print(failed_filenames)

# Convert the arrays to numpy arrays so they plot easily
val_batch_error = np.asarray(val_batch_error)
trn_batch_error = np.asarray(trn_batch_error)

# train_averaged = np.mean(trn_batch_error.reshape(-1, int(iterations / 10)), axis=1)
# valid_averaged = np.mean(val_batch_error.reshape(-1, int(iterations / 10)), axis=1)

train_averaged = []
valid_averaged = []
num_to_avg = int(iterations / 10)
for i in range(0, len(trn_batch_error)):
    max_val = i + num_to_avg
    if max_val >= len(trn_batch_error):
        max_val = len(trn_batch_error)

    train_averaged.append(np.mean(trn_batch_error[i:max_val]))
    valid_averaged.append(np.mean(val_batch_error[i:max_val]))

train_averaged = np.asarray(train_averaged)
valid_averaged = np.asarray(valid_averaged)


plt.plot(val_batch_error, 'g', label="Validation Error %")
plt.plot(trn_batch_error, 'r', label="Training Error %")
plt.plot(train_averaged, 'm', label="avg trn Error %")
plt.plot(valid_averaged, 'b', label="avg val Error %")
plt.xlabel('Batch Iterations')
plt.ylabel('Percent Error')
plt.title('Training vs Validation error rate')
plt.legend()
plt.savefig("errorpercent_" + str(learn) + "_learn_" + str(iterations) + "_iter_" + str(batch_size) + "_batchsize.png")
plt.show()


tbe_avg = np.asarray(tbe_avg)
vbe_avg = np.asarray(vbe_avg)
plt.plot(vbe_avg, 'b', label="avg val Error %")
plt.plot(tbe_avg, 'm', label="avg trn Error %")
plt.xlabel('Batch Iterations')
plt.ylabel('Percent Error')
plt.title('Training vs Validation error rate (Averaged)')
plt.legend()
plt.savefig("errorpercentaverage_" + str(learn) + "_learn_" + str(iterations) + "_iter_" + str(batch_size) + "_batchsize.png")
plt.show()
