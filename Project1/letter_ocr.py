import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ann
#np.set_printoptions(threshold=np.nan)

# Load the file and define shape information
data = pd.read_csv('./Letters.csv', header=None)
rows = data.shape[0]
cols = data.shape[1]
split = int(rows * 0.80)
print(split)

data = data.values

# Rip off the X values (Features)
X = np.array(data[:, (range(1, cols))], dtype=float)

# Normalizing each feature in X independently from the other features
for i in range(0, X.shape[1]):
    X[:, i] = (X[:, i] - X[:, i].min()) / (float(X[:, i].max()) - X[:, i].min())

# adding the bias on the input
X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

# Rip the Y's off the first column, which correspend to the letter value
Y_array = data[:, 0]

# Identify the maximum number of output nodes based on the max possible outcomes
# Assumes values are 1-N, if they are 0-(N-1) this will produce the wrong shape
output_nodes = np.max(Y_array)

# Create an array with 0.1 for "NO" and 0.9 for "YES" for each of the output nodes
# I.E. array of length 26 for letters A-Z, where 0.9 corresponds to the actual Letter index
Y = np.full([Y_array.shape[0], output_nodes], 0.1)
for i in range(Y_array.shape[0]):
    Y[i, Y_array[i] - 1] = 0.9


def calculate_accuracy(epochs, rows_in_epoch, wrong):
    total_rows = rows_in_epoch * epochs
    percent_wrong = (wrong / total_rows) * 100
    return 100 - percent_wrong


def print_stats(type, epochs, rows_in_epoch, wrong, mode, static):
    total_rows = rows_in_epoch * epochs
    percent_wrong = (wrong / total_rows) * 100
    if mode != 'simple':
        print()
        print(static, "MODE: ", type)
        print(static, "Epoch size:", epochs)
        print(static, "Rows in Epoch:", rows_in_epoch)
        print(static, "Number Wrong:", wrong)
        print(static, "Number possible:", total_rows)
        print(static, "Error percent:", percent_wrong,"%")

    print(static, "Accuracy percent:", (100 - percent_wrong),"%")
    return 100 - percent_wrong


def validate(runNum, start_pos, validation_range):
    # Quick validation at the next block
    validation_errors = 0
    for i in range(start_pos, start_pos + validation_range):
        # Fix the issue with Numpy array being (17,) instead of (1,17)
        x = X[i].reshape(X[i].shape[0], 1).T
        y = Y[i].reshape(Y[i].shape[0], 1).T

        yhat = nn.foward(x)  # 1 row at a time

        # Get the max array index for the 0-25 array (What letter)
        y_letter = np.argmax(y)
        yhat_letter = np.argmax(yhat)

        # Store the values so we can create a 2D heat map
        actual_vs_predicted[y_letter, yhat_letter] += 1

        # If we were wrong, calculate that
        if y_letter != yhat_letter:
            validation_errors += 1

    print_stats('Validation', 1, validation_range, validation_errors, "simple", runNum)
    return calculate_accuracy(1, validation_range, validation_errors)


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
    x = xin[range(batch_start, batch_start + batch_size), :]
    y = yin[range(batch_start, batch_start + batch_size), :]

    #x = xin[batch_start].reshape(xin[batch_start].shape[0], 1).T
    #y = yin[batch_start].reshape(yin[batch_start].shape[0], 1).T

    yhat = nn.foward(x)  # 1 row at a time
    nn.back_propagation(x, y)  # Includes weight updates

    # Get the max array index for the 0-25 array (What letter)
    y_letter = np.argmax(y, axis=1)
    yhat_letter = np.argmax(yhat, axis=1)
    wrong_sum = y_letter - yhat_letter
    total_wrong = (wrong_sum != 0).sum()

    return total_wrong


def run_epochs(epochs, xin, yin, batch_start, batch_size, run_validations):
    total_overall_errors = 0
    for e in range(epochs):
        total_overall_errors += run_batches(xin, yin, batch_start, batch_size)
        error_vs_epoch.append(100-validate(str(e), split, rows - split)) if run_validations else None

    print_stats('Training', epochs, len(xin), total_overall_errors, "simple", "")

    return total_overall_errors


def find_optimal_hidden_layer(epochs, xin, yin, batch_start, batch_size, min_layer_size, max_layer_size):
    global nn

    outcomes = []  # [hiddenLayers][accuracyPercent]
    input_nodes = 17
    output_nodes = 26
    tests_per_value = 3 # How many times to test the same results to average the results

    # Loop from input node count to output node count
    for nodes in range(min_layer_size, max_layer_size):
        print("")
        print("Processing with", nodes, "hidden nodes")
        avg_acc = 0.0
        for i in range(0, tests_per_value):
            print("  Test# ", i)
            nn = ann.ANN(input_nodes, nodes, output_nodes)
            run_epochs(epochs, xin, yin, batch_start, batch_size, False)

            # Test against the validation set of 4000
            avg_acc += validate("  Nodes: " + str(nodes) + " - test: " + str(i), split, rows - split)

        outcomes.append((nodes, avg_acc / tests_per_value)) # store the averaged results incase weights negatively or positively overly affected it.

    return outcomes


# Use the global values ofr learn, iterations, and batch_size to generate a related filename prefix
def make_data_filename(type_of_file, filename_part, learn_in, epochs_in, batch_size_in):
    return "./runs/" + str(learn_in) + "_learn_" \
           + str(epochs_in) + "_epochs_" \
           + str(batch_size_in) + "_batch_" \
           + str(filename_part) + "." + str(type_of_file)


def show_error_plot(errors_array, flip=False, show_or_save="save"):
    # Change to "Accuracy" plot instead
    if flip:
        errors_array = 100 - errors_array
        plt.title("Accuracy increase over epochs")
    else:
        plt.title("Error reduction over epochs")

    plt.xlabel('Epochs')
    plt.ylabel('Error Percent')

    #plt.text(len(errors_array), errors_array[-1], str(errors_array[-1]))
    plt.annotate("Final Error: " + "{0:.1f}".format(errors_array[-1]) + "%",
                 xy=(len(errors_array), errors_array[-1]),
                 xytext=(int(len(errors_array) / 2), 50),
                 arrowprops=dict(facecolor='black', shrink=0.05), )
    plt.plot(errors_array)

    if show_or_save == "save":
        if flip:
            filename = make_data_filename('png', '_accuracy_vs_epoch', nn.learn, epochs, batch_size)
        else:
            filename = make_data_filename('png', '_error_vs_epoch', nn.learn, epochs, batch_size)

        plt.savefig(filename, dpi=600)
    else:
        plt.show()

    plt.close()


def create_confusion_matrix(data, show_or_save="save"):
    # Create the "heat map"
    plt.matshow(np.asarray(data), cmap='gray', interpolation='nearest')

    # Force the tick marks to be the letters A-Z
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    plt.xticks(np.arange(len(letters)), letters)
    plt.yticks(np.arange(len(letters)), letters)

    if show_or_save == "save":
        filename = make_data_filename('png', '_confusion_matrix', nn.learn, epochs, batch_size)
        plt.savefig(filename, dpi=600)
    else:
        plt.show()

    plt.close()


epochs = 50
batch_size = 100
finding_optimal = False
actual_vs_predicted = np.zeros((26, 26))  # Create a 26x26 array for our heatmap
error_vs_epoch = []  # Create epoch vs error array

print("Creating Neural Network")
nn = ann.ANN(X.shape[1], 100, output_nodes)
print("Using Learn Rate:", nn.learn)

# Run the first 16000 rows
run_epochs(epochs, X[range(0, split), :], Y[range(0, split), :], 0, batch_size, True)

if finding_optimal:
    layer_results = find_optimal_hidden_layer(epochs, X[range(0, split), :], Y[range(0, split), :], 0, batch_size, 1, 100)
    layer_results = np.asarray(layer_results)
    print(layer_results)
    print("Best hidden Layer Size:", np.argmax(layer_results[:, -1], axis=0))

# Create a 26x26 array for our heatmap
# Run 1 last validation to populate it, Generate a plot, and save it
actual_vs_predicted = np.zeros((26, 26))
final_accuracy = validate("end", split, rows - split)
create_confusion_matrix(actual_vs_predicted)

# Convert to NP array so we can do easy matrix subtraction to convert the accuracy to error
error_vs_epoch = np.asarray(error_vs_epoch)
error_vs_epoch = 100 - error_vs_epoch

# Save the error and accuracy plots
show_error_plot(error_vs_epoch)  # Error vs epoch
show_error_plot(error_vs_epoch, True)  # Accuracy vs epoch (Flipped error value to accuracy)
