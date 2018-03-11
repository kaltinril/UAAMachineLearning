

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
    working_row = csv_row.split(',')  # Convert the CSV row into an array
    # Remove the first row (prediction) and the last row(filename)
    prediction = str(working_row[0])
    filename = "# " + str(working_row[-1])
    working_row = working_row[1:-1]

    # Convert 0 to -1 in prediction, add a space after
    if prediction == '0':
        prediction = '-1 '
    else:
        prediction = prediction + " "

    # Build the SVM formatted row
    working_row = build_feature_value(working_row)

    # Combine it all back together
    svm_row = [prediction] + working_row + [filename]

    return svm_row


def convert_file(input_filename, output_filename):
    # Open Source and Destination files
    input_file = open(input_filename, 'r')
    output_file = open(output_filename, 'w')

    # Loop over rows
    for row in input_file:
        # build svm formatted row
        svm_row = convert_row(row)

        # Save row to file
        output_file.writelines(svm_row)

    # Close Files
    input_file.close()
    output_file.close()


def main():
    convert_file('aurora_histogram.csv', 'ah.svm')


if __name__ == "__main__":
    main()
