import cv2
import os
import sys

# Optionally this could be an input argument to the script
output_filename = './aurora_histogram.csv'


def extract_histograms(directory, aurora, output_handle):

    # Load images from folders in loop
    for filename in os.listdir(directory):
        combined_filename = os.path.join(directory, filename)
        img = cv2.imread(combined_filename)
        series_str = str(aurora)

        # Loop over the three colors (Blue, Green, Red) (OpenCV has this order)
        for i in range(0, 3):
            try:
                series = cv2.calcHist([img], [i], None, [256], [0, 256])
            except:
                print("Failed on file: " + str(combined_filename))
                print("Unexpected error:", sys.exc_info()[0])

            # Brute force convert the numpy array of arrays of doubles to a CSV format
            for value in series:
                series_str = series_str + "," + str(int(value[0]))

        # Add a new line for the image
        series_str = series_str + "," + combined_filename + '\n'

        # Write the histogram, with the Aurora value and filename out to the file
        output_handle.writelines(series_str)


def main():
    output_file = open(output_filename, 'w')
    extract_histograms('./images/aurora', 1, output_file)
    extract_histograms('./images/none', 0, output_file)
    output_file.close()


if __name__ == "__main__":
    main()
