import numpy as np
import cv2


def create_predictions(data_shape):
    predictions = np.zeros((data_shape[0], 3), dtype=int)

    for i in range(0, predictions.shape[0]):
        a = int((i // (63 * 8 * 60))) + 1
        p = ((int((i // (63 * 60)))) % 8) + 1
        s = (i % 60) + 1

        predictions[i, 0] = int(a)
        predictions[i, 1] = int(p)
        predictions[i, 2] = int(s)

    return predictions

def normalize_data(data):

    for col in range(0, data.shape[1]):
        column_data = data[:, col:col+1]
        column_data = (column_data - column_data.min()) / (float(column_data.max()) - column_data.min())
        data[:, col:col+1] = column_data

    return data



# Load the PCA'd data
pca = np.load('./pca.npy')
#predictions = create_predictions(pca.shape)

# Normalize it
pca = normalize_data(pca)
print(pca.shape)

k = 3

# create n centroides by picking random values in each feature range (column)
centroids = np.random.uniform(0, 1, (k, pca.shape[1]))

print(centroids.shape)


# assign
result = []
for cent in range(0, centroids.shape[0]):
    subtracted = pca - cent
    normalized = np.linalg.norm(subtracted, axis=1)
    result.append(normalized)
    print(normalized.shape)

final_result = np.vstack(result)
print(final_result.shape)

assignments = np.argmax(final_result, axis=0)
print(assignments.shape)

