import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

# plt.figure()
# plt.xlim([0, 256])
# plt.ylim([0, 20000])

color = ('b', 'g', 'r')

output_filename = './aurora_hist1.csv'
output_handle = open(output_filename, 'w')

complete_hist = pd.DataFrame()

# Load images from folders in loop
directory = '../unknown'
for filename in os.listdir(directory):
    combined_filename = os.path.join(directory, filename)
    img = cv2.imread(combined_filename)

    aurora = '1'
    series_str = aurora

    for i, col in enumerate(color):
        series = cv2.calcHist([img], [i], None, [256], [0, 256])
        # plt.plot(series, color=col)

        # Brute force convert the numpy array of arrays of doubles to a CSV format
        for value in series:
            series_str = series_str + "," + str(int(value[0]))

    series_str = series_str + "," + combined_filename + '\n'
    output_handle.writelines(series_str)

output_handle.close()



