import cv2
import os

output_filename = './aurora_hist1.csv'
output_handle = open(output_filename, 'w')

# Load images from folders in loop
directory = '../unknown'
aurora = '1'
for filename in os.listdir(directory):
    combined_filename = os.path.join(directory, filename)
    img = cv2.imread(combined_filename)
    series_str = aurora

    # Loop over the three colors (Blue, Green, Red) (OpenCV has this order)
    for i in range(0, 3):
        series = cv2.calcHist([img], [i], None, [256], [0, 256])

        # Brute force convert the numpy array of arrays of doubles to a CSV format
        for value in series:
            series_str = series_str + "," + str(int(value[0]))

    series_str = series_str + "," + combined_filename + '\n'
    output_handle.writelines(series_str)

output_handle.close()
