import matplotlib.pyplot as plt
from plotData import PlotData
from random import *
import sys

# Load values from command line
filename = sys.argv[1]

# Seems that if we use the complete data-set too many times, we "overtrain" it and get bad results (10)
iterations = int(sys.argv[2])

# Print out the values used
print("Using: filename=" + filename + " iterations=" + str(iterations))

myData = PlotData()
myData.load_data(filename)

# Dynamically add labels, assuming the 2nd column is the X, and the third is the Y
plt.xlabel(myData.headers[1])
plt.ylabel(myData.headers[2])
plt.title('Linear Regression: ' + myData.headers[1] + " vs " + myData.headers[2])

b1 = 1.0
b0 = 1.0
batchStart = 0  # Always start at 0
learn_rate = 0.05 # Learn rate is high because we've normalized the dataset, otherwise a super low rate 0.000000001 is needed


# Define a method for returning the y value based on X, b1, and b0
def y(x, b_0, b_1):
    return b_1*x + b_0


def gradient_decent(X, Y, batch_start, batch_size, b_0, b_1):
    total_diff0 = 0
    total_diff1 = 0

    for i in range(batch_start, batch_start + batch_size):
        total_diff0 += y(X[i], b_0, b_1) - Y[i]
        total_diff1 += (y(X[i], b_0, b_1) - Y[i]) * X[i]

    return total_diff0/batch_size, total_diff1/batch_size


# Run through the dataset a batch_size at a time
# Doing the gradient decent to hone in on the
# best b1 and b0
def batch_looping(b_0, b_1, batch_start, batch_size):
    while batch_start < len(myData.X):
        if batch_start + batch_size > len(myData.X):
            batch_size = len(myData.X) - batch_start
        td1, td2 = gradient_decent(myData.X, myData.Y, batch_start, batch_size, b_0, b_1)

        # Update the b0 and b1 values by the learn rate times the total difference
        b_1 = b_1 - learn_rate * td2
        b_0 = b_0 - learn_rate * td1

        print("b0: " + str(b_0) + " b1: " + str(b_1) + " error: " + str(td2))

        # Increment the starting position by the batch size
        batch_start += batch_size

    return b_0, b_1


def iterate_and_plot(iters, b_0, b_1, batch_start, batch_size):
    # Loop over the dataset as many iterations as we want
    for i in range(iters):
        b_0, b_1 = batch_looping(b_0, b_1, batch_start, batch_size)

    # Plot the line using the new b1 and b0 values
    plt.plot(myData.X, b_1 * myData.X + b_0, label='Batch Size: ' + str(batch_size))

    # Pick 5 random values and plot them
    #predice_points(b_0, b_1)

    print("Iteration with batchsize: " + str(batch_size))

    return b_0, b_1


def predice_points(b_0, b_1):
    new_x = [random(), random(), random(), random(), random()]
    new_y = [y(new_x[0], b_0, b_1),
             y(new_x[1], b_0, b_1),
             y(new_x[2], b_0, b_1),
             y(new_x[3], b_0, b_1),
             y(new_x[4], b_0, b_1)]

    plt.plot(new_x, new_y, 'ro')


# Per the assignment, we are to do batches of 1, 54, 10, 15, and the complete dataset
iterate_and_plot(iterations, b0, b1, batchStart, 1)
iterate_and_plot(iterations, b0, b1, batchStart, 5)
iterate_and_plot(iterations, b0, b1, batchStart, 10)
iterate_and_plot(iterations, b0, b1, batchStart, 15)
iterate_and_plot(iterations, b0, b1, batchStart, len(myData.X))

# Display all the points last so they are on-top of the lines
plt.plot(myData.X, myData.Y, 'bo')

# Because we added labels to the lines, show a legend
plt.legend()

filename_prefix = filename.split('-', 1)[0]
plt.savefig(filename_prefix + "_" + str(iterations) + "_iter.png")

# For windows the plot won't show up unless you include this
plt.show()


