import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

DEBUG = True


def run_pca(data):
    # Remove the first row (prediction) and the last row(filename)
    predictions = data[:, 0].astype(str)    # Get the predictions out of the array
    filenames = data[:, -1]                 # Get the filenames off the end of the array
    working_data = data[:, 1:-1]            #.astype(float)    # Only snag rows 1 to last row

    # run it through PCA
    pca = PCA(.95)                          # Only keep first N features that add up to cumulative 95% of the variance
    pca.fit(working_data)                   # Fit the data
    data_pca = pca.transform(working_data)  # Transform the data

    # Add the predictions and the filenames back onto the array
    data_pca = np.c_[predictions, data_pca, filenames]

    return data_pca


def main():
    input_filename = "./aurora_histogram.csv"
    output_filename = "./aurora_hist_pca.csv"

    # Load file
    data = pd.read_csv(input_filename, header=None)
    data = data.values

    # Run the PCA on the input to remove features (columns) that are not significant
    transformed_data = run_pca(data)
    np.savetxt(output_filename, transformed_data, fmt="%s", delimiter=',')


if __name__ == "__main__":
    main()
