import numpy as np

'''
csv_to_svm_converter
    Take a CSV formatted list of features, where the first row is the prediction and the last row is the filename,
    and convert to the SVM-Light required format.

    Created by:     Jeremy Swartwood
input:
        <prediction>,<value1>,<value2,...<valuen>,<filename>
        1,4,767,....,2323,c:\path\to\filename1.png
        0,6,823,....,3623,c:\path\to\filename2.png
output:
        <prediction> <feature>:<value> <feature>:<value> ... <feature>:<value> # <filename>
        Where Feature is an incrementing number for each feature
        Value is the value pulled from the input CSV file in order

        1 1:4 2:767 ... 768:2323 # c:\path\to\filename1.png
       -1 1:5 2:823 ... 768:4623 # c:\path\to\filename2.png
'''


def build_feature_value(row):
    feature_value_row = []
    i = 0
    for value in row:
        i += 1

        # Don't add values that are 0, per the documentation these can be left out of the file
        if value != 0:
            feature_value_row.append(str(i) + ":" + str(value) + " ")  # Add space to seperate the values

    return feature_value_row


def convert_row(csv_row):
    # Remove the first row (prediction) and the last row(filename)
    prediction = str(csv_row[0])
    filename = "# " + str(csv_row[-1])
    working_row = csv_row[1:-1]

    # Convert 0 to -1 in prediction, add a space after either
    if prediction == '0':
        prediction = '-1 '
    else:
        prediction = prediction + " "

    # Build the SVM formatted row
    working_row = build_feature_value(working_row)

    # Combine it all back together
    svm_row = [prediction] + working_row + [filename] + ['\n']

    return svm_row


def convert_file(input_filename, training_filename, validation_filename, split_percent=0.80):
    # Open Source and Destination files
    print("Loading source input file")
    input_file = np.genfromtxt(input_filename, delimiter=',', dtype=str)
    training_file = open(training_filename, 'w')
    validation_file = open(validation_filename, 'w')

    # Randomize the data input
    print("Shuffling the data")
    np.random.shuffle(input_file)

    # Split the file into training and validation
    print("Splitting the data into training and validation arrays")
    split_point = int(len(input_file) * split_percent)
    training_input = input_file[0:split_point, :]
    validation_input = input_file[split_point:len(input_file), :]

    # Loop over Training rows
    print("Converting Training Rows")
    for row in training_input:
        svm_row = convert_row(row)  # build svm formatted row
        training_file.writelines(svm_row)  # Save row to file

    # Loop over Validation rows
    print("Converting Validation Rows")
    for row in validation_input:
        svm_row = convert_row(row)  # build svm formatted row
        validation_file.writelines(svm_row)  # Save row to file

    # Close File
    print("Cleaning up, closing files")
    training_file.close()
    validation_file.close()


def main():
    convert_file('aurora_histogram.csv', 'ah_training.svm', 'ah_validate.svm')


if __name__ == "__main__":
    main()
