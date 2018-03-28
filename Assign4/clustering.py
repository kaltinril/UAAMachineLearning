import numpy as np
import pandas as pd
import time




# Loop over all the activities, all the people, and all the samples and load the data.
data = None
y = None
for activity in range(1, 20):
    start = time.clock()
    print("Activity:", activity)
    for person in range(1, 9):
        for sample in range(1, 61):
            row_data = pd.read_csv('./ClusteringData/data/a'
                                   + format(activity, '02d')
                                   + '/p' + str(person) + '/s'
                                   + format(sample, '02d')
                                   + '.txt', header=None).values

            row_y = [activity, person, sample]

            if data is None:
                data = row_data
                y = row_y
            else:
                data = np.vstack((data, row_data))
                y = np.vstack((y, row_y))

    end = time.clock()
    print("Elapsed:", end - start)


print(data.shape)
