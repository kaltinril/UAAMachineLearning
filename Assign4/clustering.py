import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


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


def load_data(filename='./SyntheticData.txt', type='txt'):
    # Load the PCA'd data

    if type == 'txt':
        data = np.loadtxt(filename)
    elif type == 'npy':
        data = np.load(filename)
    else:
        print('ERROR: Unknown file type! ', type)
        sys.exit(1)

    # Normalize it
    return normalize_data(data)


def adjust_cluster_center(data, centroids, assignments):
    for cent in range(0, centroids.shape[0]):

        mask = (assignments == cent)
        cluster = data[mask]

        if cluster is not None and len(cluster) > 0:
            centroids[cent, :] = np.mean(cluster, axis=0)
        else:
            # No points, re-assign the centroid
            centroids[cent, :] = np.random.uniform(0, 1, (1, data.shape[1]))





def assign_clusters(data, centroids):
    # assign
    result = []
    for cent in range(0, centroids.shape[0]):
        subtracted = data - centroids[cent, :]
        normalized = np.linalg.norm(subtracted, axis=1)
        result.append(normalized)

    final_result = np.vstack(result)
    print('final_result', final_result.shape)

    assignments = np.argmin(final_result, axis=0)
    print(assignments.shape)

    return assignments


def display_results(data, centroids, assignments):
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='rainbow')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')
    plt.show()

def run_clustering(data, k=2):
    # create n centroides by picking random values in each feature range (column)
    centroids = np.random.uniform(0, 1, (k, data.shape[1]))
    print(centroids.shape)

    assignments = []
    for i in range(0, 10):
        assignments = assign_clusters(data, centroids)
        adjust_cluster_center(data, centroids, assignments)

    display_results(data, centroids, assignments)

loaded_data = load_data(filename='./SyntheticData.txt', type='txt')
run_clustering(loaded_data, 8)


