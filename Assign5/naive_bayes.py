import numpy as np
import os
import time
import matplotlib.pyplot

column_prediction = 0

def load_data(filename='./SpamInstances.txt'):
    # Load the entire file into memory
    file_data = np.loadtxt(filename, delimiter=' ', skiprows=1, dtype=str)

    # Because the third "column" is actually 334 features/columns, we need to split them
    new_data = []
    for i in range(file_data.shape[0]):
        new_data.append(np.array(list(str(file_data[i, 2]))))

    # Convert the 3rd column (that is now 334 columns) into a numpy array of floats
    new_data = np.vstack(new_data).astype(np.float)

    # Slap the 334 columns onto the first 2 columns
    return np.hstack((file_data[:, 1:2].astype(np.int), new_data))


def split_yes_no(data):
    # Split into YES spam and NO spam
    yes_mask = (data[:, column_prediction] == 1)  # Return TRUE/FALSE for all rows where column 1 (spam/not) is 1 (Spam)
    yes = data[yes_mask]  # Run the TRUE/FALSE mask to only return rows where TRUE is set (Only the "SPAM" rows)

    no_mask = (data[:, column_prediction] == -1)  # Return TRUE/FALSE for all rows where column 1 (spam/not) is -1 (Not spam)
    no = data[no_mask]  # Run the TRUE/FALSE mask to only return rows where TRUE is set (Only the "NON-SPAM" rows)

    return yes, no


def split_training_validation(all_data, number=0, percent=1):
    yes, no = split_yes_no(all_data)

    # How many rows total?
    if percent!= 1:
        number = all_data.shape[0] * percent

    # How many yes and no do we get?
    num_yes = int(number / 2)
    num_no = int(number - num_yes) # If odd, make sure we don't go over the number value

    # Randomize them (Should we do this?  Maybe only when the program starts?)
    #np.random.shuffle(yes)
    #np.random.shuffle(no)

    yes_rows = yes[0:num_yes, :]
    no_rows = no[0:num_no, :]
    training = np.vstack((yes_rows, no_rows))

    yes_rows = yes[num_yes:, :]
    no_rows = no[num_no:, :]
    validation = np.vstack((yes_rows, no_rows))

    return training, validation


def create_column_probabilities(training):
    yes, no = split_yes_no(training)

    # Remove the prediction column, splitting it up did that
    yes = yes[:, 1:]
    no = no[:, 1:]

    # sum across all to get a single row of sums for no and yes
    yes_sum = np.sum(yes, axis=0) + 1
    no_sum = np.sum(no, axis=0) + 1

    # Divide by the total "SPAM" or "NON_SPAM" counts
    print('Total SPAM:', yes.shape[0])
    print('Total Non:', no.shape[0])
    yes_prob = yes_sum / (yes.shape[0] + 1)
    no_prob = no_sum / (no.shape[0] + 1)

    return yes_prob, no_prob


def predict_one(row, yes, no, total_rows):

    yes_prob = []
    no_prob = []
    for i in range(len(row)):

        yes_prob.append(yes[int(row[i]), i])
        no_prob.append(no[int(row[i]), i])
        #print(row[i])

    yes_prob = np.hstack(yes_prob)
    no_prob = np.hstack(no_prob)

    # Do the log of each
    yes_prob = np.log2(yes_prob)
    no_prob = np.log2(no_prob)

    # Sum the values up, multiply by the YES / TOTAL and NO / TOTAL
    yes_prob = np.sum(yes_prob) * (yes.shape[1] / total_rows)
    no_prob = np.sum(no_prob) * (no.shape[1] / total_rows)

    #print(yes_prob)
    #print(no_prob)

    result = -1 # assume not spam
    if yes_prob > no_prob:
        result = 1

    return result


def predict(validation, train, yes, no):
    results = []
    correct = 0
    for row in range(validation.shape[0]):
        result = predict_one(validation[row, 1:], yes, no, train.shape[0])
        if result == validation[row, 0:1]:
            correct += 1

        results.append(result)

    return correct


def run_all(data, num=0, per=1):

    train, valid = split_training_validation(data, number=num, percent=per)

    yes_prob, no_prob = create_column_probabilities(train)

    # Each column is 0 or 1, so lets now find the "0"'s
    yes_flip_prob = 1 - yes_prob
    no_flip_prob = 1 - no_prob

    # Add 1 to prevernt nan and underflow/overflow/divide-by-zero
    # yes_prob = yes_prob + 1
    # no_prob = no_prob + 1
    # yes_flip_prob = yes_flip_prob + 1
    # no_flip_prob = no_flip_prob + 1

    yes = np.vstack((yes_flip_prob, yes_prob))
    no = np.vstack((no_flip_prob, no_prob))

    correct = predict(valid, train, yes, no)

    print('Correct:', correct, ' Percent ', correct / valid.shape[0])

    return correct / valid.shape[0]


data = load_data()

results = []
for i in range(20):
    results.append(run_all(data, num=(i+1)*100))

results.append(run_all(data, per=0.80))

print(results)
