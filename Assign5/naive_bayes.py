import numpy as np
import time
import matplotlib.pyplot as plt
import sys

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


# Take in a dataset and split the values into all the SPAM [1] vs non-spam [-1]
def split_yes_no(data):
    # Split into YES spam and NO spam
    yes_mask = (data[:, column_prediction] == 1)  # Return TRUE/FALSE for all rows where column 1 (spam/not) is 1 (Spam)
    yes = data[yes_mask]  # Run the TRUE/FALSE mask to only return rows where TRUE is set (Only the "SPAM" rows)

    no_mask = (data[:, column_prediction] == -1)  # Return TRUE/FALSE for all rows where column 1 (spam/not) is -1 (Not spam)
    no = data[no_mask]  # Run the TRUE/FALSE mask to only return rows where TRUE is set (Only the "NON-SPAM" rows)

    return yes, no


# split the dataset into TRAINING and VALIDATION datasets of equal portions from the YES(spam) and NO(not) categories
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


# Generate the probabilities for each of the 334 features
# Since we can assume no conditional probability between features, this simplifies everything
def create_column_probabilities(training):
    yes, no = split_yes_no(training)

    # Remove the prediction column, splitting it up did that
    yes = yes[:, 1:]
    no = no[:, 1:]

    # sum across all to get a single row of sums for no and yes
    yes_sum = np.sum(yes, axis=0) + 1
    no_sum = np.sum(no, axis=0) + 1

    # Divide by the total "SPAM" or "NON_SPAM" counts
    print('Total Training Data SPAM:', yes.shape[0], ' Non-Spam:', no.shape[0])
    yes_prob = yes_sum / (yes.shape[0] + 1)
    no_prob = no_sum / (no.shape[0] + 1)

    # The number of rows in the YES dataset
    yes_count = yes.shape[0]
    no_count = no.shape[0]

    return yes_prob, no_prob, yes_count, no_count


def predict_one(row, yes, no, yes_count, no_count, total_rows):

    # Create index lookup of all the "1" values and all the "0" values
    ones = np.where(row == 1)[0]
    zeros = np.where(row == 0)[0]

    # Use the lookup values to find the YES and NO probabilities for this row
    # Luckily, we don't care about the order since its all going to be added together.
    yes_prob = np.hstack((yes[1, ones], yes[0, zeros]))
    no_prob = np.hstack((no[1, ones], no[0, zeros]))

    # Do the log of each instance so we don't get numerical underflow or overflow
    yes_prob = np.log(yes_prob)
    no_prob = np.log(no_prob)

    # "The log of the products, is the sum of the logs."
    # Sum the values up, multiply by the (YES / TOTAL) and (NO / TOTAL)
    yes_prob = np.sum(yes_prob) + np.log2(yes_count / total_rows)
    no_prob = np.sum(no_prob) + np.log2(no_count / total_rows)

    # do a "argmax" to find out which probability is greater
    result = -1  # assume not spam
    if yes_prob > no_prob:
        result = 1  # Spam

    return result


def predict(validation, train, yes, no, yes_count, no_count):
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for row in range(validation.shape[0]):
        result = predict_one(validation[row, 1:], yes, no, yes_count, no_count, train.shape[0])
        if result == validation[row, 0:1]:
            correct += 1
            if result == 1:
                tp += 1
            else:
                tn += 1
        else:
            if result == 1:
                fp += 1
            else:
                fn += 1

    return correct, tp, tn, fp, fn


def run_all(data, num=0, per=1):
    start = time.time()
    train, valid = split_training_validation(data, number=num, percent=per)

    yes_prob, no_prob, yes_count, no_count = create_column_probabilities(train)

    # Each column is 0 or 1, so lets now find the "0"'s
    yes_flip_prob = 1 - yes_prob
    no_flip_prob = 1 - no_prob

    # Stack the SPAM where each feature is 1 with the SPAM where each feature is 0 vertically
    yes = np.vstack((yes_flip_prob, yes_prob))
    no = np.vstack((no_flip_prob, no_prob))  # Do the same for the NON-Spam

    correct, tp, tn, fp, fn = predict(valid, train, yes, no, yes_count, no_count)

    print('Validation instances:', valid.shape[0], ' Correct:', correct, ' Percent ', correct / valid.shape[0], ' Run Time', time.time() - start)

    return correct / valid.shape[0], tp, tn, fp, fn


def generate_roc_curve(results):
    # Calculate the FPR and TPR from the tp=0, tn=1, fp=2, fn=3 values.
    fpr = results[:, 2] / np.sum(results[:, (2, 1)], axis=1)  # FP / (FP + TN)
    tpr = results[:, 0] / np.sum(results[:, (3, 0)], axis=1)  # TP / (FN + TP)

    # Show the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title("ROC Space")
    plt.show()


def show_accuracy(accuracy):
    plt.plot(accuracy)
    plt.show()
    plt.clf()


def main(argv):
    print("Naive Bayes - SPAM classifier")
    print("")

    print('Loading Data')
    data = load_data()

    print('Running 20 times')
    # Run starting at 100 size, re-run 20 times incrementing by 100 each time
    results = []
    for i in range(20):
        results.append(run_all(data, num=(i + 1) * 100))

    # Run for 80% = Training || 20% = Validation
    results.append(run_all(data, per=0.80))

    # Convert array to numpy array
    results = np.array(results)

    # Show the accuracy (plotted)
    show_accuracy(results[:, 0])

    # Send columns 1-4 over, (tp, tn, fp, fn) skipping first column (accuracy)
    generate_roc_curve(results[:, 1:])

    print("")
    print("Successfully completed analysis")

    return results


if __name__ == "__main__":
    main(sys.argv)
