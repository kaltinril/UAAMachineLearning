import numpy as np
import time

print("Building Filename List")
start = time.clock()
data = None
y = None
fnames = []
for activity in range(1, 20):
    for person in range(1, 9):
        for sample in range(1, 61):
            fnames.append('./ClusteringData/data/a'
                          + format(activity, '02d')
                          + '/p' + str(person) + '/s'
                          + format(sample, '02d')
                          + '.txt')

end = time.clock()
print("Elapsed Filename Building:", end - start)
print('')

print("Loading data")
start = time.clock()
arrays = [np.loadtxt(f, delimiter=',') for f in fnames]
end = time.clock()
print("Elapsed Loading Time:", end - start)
print('')

print('Concatenating files data into a single array')
start = time.clock()
final_array = np.concatenate(arrays)
end = time.clock()
print("Elapsed Concatenating Time:", end - start)
print('')

print(final_array.shape)

np.save('./data_combined', final_array)

fnames = None
arrays = None
final_array = None
