import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys

print("Aurora photo analysis using Machine Learning Perceptron and gradient batch decent")
start_time = time.time()

DEBUG = True
random.seed()

# Load parameters from command line
try:
    learn = float(sys.argv[1])
    iterations = int(sys.argv[2])
    batch_size = int(sys.argv[3])
except:
    print("Error, missing one of the three required parameters.")
    print("usage: preceptron.py <learn> <iterations> <batch_size>")
    exit(1)

# Print out the values found in the arguments
print("Using: learn=" + str(learn) + " iterations=" + str(iterations) + " batch_size=" + str(batch_size))

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

# Convert the arrays to numpy arrays for easier manipulation
aurora = np.asarray(aurora)
not_aurora = np.asarray(not_aurora)

# Shuffle the aurora and not aurora arrays, so the order is random
print("DEBUG: Randomize the order of the AURORA and NOT AURORA rows") if DEBUG else None
np.random.shuffle(aurora)
np.random.shuffle(not_aurora)

# Split the arrays into 80% and 20% for training and validation
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

# Shove the arrays ontop since the code uses one array and picks out the 80%
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
    W[i] = random.randint(-1, 1)  # Set each W[i] value to a random value of -1, 0, or 1

# Normalize the data from 0-1 across the entire data-set
print("DEBUG: Normalizing the X values") if DEBUG else None
X = (X - X.min()) / (float(X.max()) - X.min())

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

print("Starting Perceptron Training and Validation batches")
print("********* Runtime parameters ***********")
print("Used Batch Size: " + str(batch_size))
print("Used Iterations: " + str(iterations))
print("Used Learn Rate: " + str(learn))
start_perceptron = time.time()


# Use the global values ofr learn, iterations, and batch_size to generate a related filename prefix
def make_data_filename(type_of_file, file_content):
    return "./runs/" + str(learn) + "_learn_" \
           + str(iterations) + "_iter_" \
           + str(batch_size) + "_batch_" \
           + str(file_content) + "." + str(type_of_file)


# This is the Y Hat
# Y is the 1 or 0 for "Aurora = 1" and "Not = 0"
def perceptron_calculation(histograms, row_index):
    x_values = histograms[row_index]

    # Calculate the Perceptron value
    # Below is doing the following: total = W0 + sum(X[i]*W[i])
    # This is the "dot product" of the two arrays
    total = np.dot(W, x_values)
    total = total + WZ

    result = 0
    if total > 0:
        result = 1

    return result


# wb = wb - m (y^ – y) * x
def adjust_weight(histograms, hist_y, row_index, y_hat):
    global WZ
    global W
    x_values = histograms[row_index]

    # Hoping the pre-computed values will help with speed
    error_value = y_hat - hist_y[row_index]
    learn_error = learn * error_value

    # Use numpy's array manipulation instead of looping over
    x_values_times_learn = np.dot(x_values, learn_error)   # learn_error * x_values[column]
    W = W - x_values_times_learn  # W[column] = W[column] - x_values_times_learn
    WZ = WZ - learn_error


train_validate_split = int(rows * .8)
tbavgcnt = 0
vbavgcnt = 0
avg_across = iterations / 10
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

    tbavgcnt = tbavgcnt % avg_across
    vbavgcnt = vbavgcnt % avg_across
    tbavgcnt += 1
    vbavgcnt += 1

    if tbavgcnt == avg_across:
        tbe_avg.append(tavgsum / avg_across)
        vbe_avg.append(vavgsum / avg_across)

end_perceptron = time.time()  # Capture timing for calculations of how long different parts took

TRN_PER = TRN / (iterations * batch_size)
ACC_PER = ACC / (iterations * batch_size)
print("")
print("********* Results ***********")
print("Training Error %: " + str(1-TRN_PER))
print("Validate Error %: " + str(1-ACC_PER))


# Save the final weights and the "WZ" value
W = np.asarray(W)
np.savetxt(make_data_filename('csv', 'batch_weights'), W, delimiter=',')
output_file = open(make_data_filename('txt', 'batch_wz'), 'w')
output_file.write(str(WZ))
output_file.close()

# Save the graph of the final weights
plt.plot(W[range(0, 255)], 'bo', label="Weights for blue")
plt.plot(W[range(256, 511)], 'go', label="Weights for green")
plt.plot(W[range(512, 768)], 'ro', label="Weights for red")
plt.xlabel('0-255 (Color range)')
plt.ylabel('Weight')
plt.title('Weight vs Color value')
plt.legend()
plt.savefig(make_data_filename('png', 'weights_plot'), dpi=600)
plt.close()

# Save the files that failed
print("Total Failed files in last batch:", len(failed_filenames))
output_file = open(make_data_filename('txt', 'batch_failed_files'), 'w')
output_file.write(str(failed_filenames))
output_file.close()

# Convert the arrays to numpy arrays so they plot easily
val_batch_error = np.asarray(val_batch_error)
trn_batch_error = np.asarray(trn_batch_error)

# Average the every (iterations / 10) values together so we get an average to smooth out the drastic changes
train_averaged = []
valid_averaged = []
num_to_avg = int(iterations / 10)
for i in range(0, len(trn_batch_error)):
    max_val = i + num_to_avg
    if max_val >= len(trn_batch_error):
        max_val = len(trn_batch_error)

    train_averaged.append(np.mean(trn_batch_error[i:max_val]))
    valid_averaged.append(np.mean(val_batch_error[i:max_val]))

# Convert the arrays to numpy arrays so they plot easily
train_averaged = np.asarray(train_averaged)
valid_averaged = np.asarray(valid_averaged)

# Graph the error rates and the averaged rate
plt.plot(val_batch_error, 'g', label="Validation Error %")
plt.plot(trn_batch_error, 'r', label="Training Error %")
plt.plot(train_averaged, 'c', label="avg trn Error %")
plt.plot(valid_averaged, 'b', label="avg val Error %")
plt.xlabel('Batch Iterations')
plt.ylabel('Percent Error')
plt.title('Training vs Validation error rate')
plt.legend()
plt.savefig(make_data_filename('png', 'batch_err_pct'), dpi=600)
plt.close()

# Exaggerated (highly averaged) graph of train vs validation
tbe_avg = np.asarray(tbe_avg)
vbe_avg = np.asarray(vbe_avg)
plt.plot(vbe_avg, 'b', label="avg val Error %")
plt.plot(tbe_avg, 'c', label="avg trn Error %")
plt.xlabel('Batch Iterations (averaged)')
plt.ylabel('Percent Error')
plt.title('Training vs Validation error rate (Averaged)')
plt.legend()
plt.savefig(make_data_filename('png', '_batch_err_pct_avg.png'), dpi=600)
plt.close()

end_program = time.time()
print("")
print("********* Runtime information **********")
print("Total runtime: " + str(end_program - start_time))
print("Perceptron Time: " + str(end_perceptron - start_perceptron))
print("Plotting: " + str(end_program - end_perceptron))
