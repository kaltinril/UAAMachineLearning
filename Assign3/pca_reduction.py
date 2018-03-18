from sklearn.decomposition import PCA
import numpy as np          # Used for image and array manipulation
import pandas as pd         # Easier to load mutli data-type matrix/array/files in
import getopt               # Friendly command line options
import sys                  # Used to get the sys.argv options

'''
pca_reduction
    Take an input file of features, predictions, and associated notes and collapse the features.
    Each row should be <prediciton "Y" value>, <one or more features>, <any notes about this row like filename>
    
    Created by:     Jeremy Swartwood
Usage:
    To use this in another python library:
        import cv2
        import pca_reduction

        mask = pca_reduction.run_pca("aurora_histogram.csv")
'''

DEBUG = True
DEFAULT_INPUT_FILENAME = "./aurora_histogram.csv"
DEFAULT_OUTPUT_FILENAME = "./aurora_hist_pca.csv"

def add_non_feature_columns(original_data, pca_data):
    # Remove the first row (prediction) and the last row(filename)
    predictions = original_data[:, 0].astype(str)    # Get the predictions out of the array
    filenames = original_data[:, -1]                 # Get the filenames off the end of the array

    # Add the predictions and the filenames back onto the array
    data_pca = np.c_[predictions, pca_data, filenames]

    return data_pca


def perform_pca_reduction(data):
    # run it through PCA
    pca = PCA(.95)                          # Only keep first N features that add up to cumulative 95% of the variance
    pca.fit(data[:, 1:-1] )                   # Fit the data
    pca_data = pca.transform(data[:, 1:-1] )  # Transform the data

    return pca_data


def run_pca(input_filename, output_filename):
    # Load data file with features, assume no header
    data = pd.read_csv(input_filename, header=None)
    data = data.values

    # Run the PCA on the input to remove features (columns) that are not significant
    pca_data = perform_pca_reduction(data[:, 1:-1])  # exclude column 0 and the last column

    # Add the prediction and filenames back
    transformed_data = add_non_feature_columns(data, pca_data)

    # Save the reduced feature data back to a file
    np.savetxt(output_filename, transformed_data, fmt="%s", delimiter=',')

    return transformed_data


def print_help(script_name):
    print("Usage:   " + script_name + " -o <output_filename> -i <input_filename>")
    print("")
    print(" -h, --help")
    print("    This message is printed only")
    print(" -o, --outfile")
    print("    Output file to save to")
    print("    default: aurora_hist_pca.csv")
    print(" -i, --infile")
    print("    Input file of features")
    print("    default: aurora_histogram.csv")
    print(" -d, --debug")
    print("    Turn debug mode on")
    print("")
    print("Example: " + script_name + ' -i aurora.csv -o converted.csv')


def load_arguments(argv):
    global DEBUG
    script_name = argv[0]  # Snag the first argument (The script name)

    # Default values for parameters/arguments
    input_filename = DEFAULT_INPUT_FILENAME
    output_filename = DEFAULT_OUTPUT_FILENAME

    # No reason to parse the options if there are none, just use the defaults
    if len(argv) > 1:
        try:
            single_character_options = "ho:i:d"  # : indicates a required value with the value
            full_word_options = ["help", "outfile=", "infile=", "debug"]

            opts, remainder = getopt.getopt(argv[1:], single_character_options, full_word_options)
        except getopt.GetoptError:
            print("ERROR: Unable to get command line arguments!")
            print_help(script_name)
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_help(script_name)
                sys.exit(0)
            elif opt in ("-o", "--outfile"):
                output_filename = arg
            elif opt in ("-i", "--infile"):
                input_filename = arg
            elif opt in ("-d", "--debug"):
                DEBUG = True  # Global variable for printing out debug information

    print("Using parameters:")
    print("Output File:     ", output_filename)
    print("Input File:      ", input_filename)
    print("Debug:           ", DEBUG)
    print("")

    return input_filename, output_filename


def main(argv):
    print("PCA reduction on input file")
    print("")

    # Load all the arguments and return them
    input_filename, output_filename = load_arguments(argv)

    # Run the PCA reduction
    transformed_data = run_pca(input_filename, output_filename)

    print("")
    print("Successfully completed, look for file", output_filename)

    return transformed_data


if __name__ == "__main__":
    main(sys.argv)
