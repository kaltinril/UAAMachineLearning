import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial


def create_predictions(data_shape):
    predictions = np.zeros((data_shape[0], 4), dtype=int)

    sedentary = [1,2,3,4,7,8]
    active = [5,6,9,10,11]
    excersize = [12,13,14,15,16,17,18,19]
    cluster = 0

    for i in range(0, predictions.shape[0]):
        a = int((i // (8 * 60))) + 1
        p = ((int((i // (60)))) % 8) + 1
        s = (i % 60) + 1

        if a in sedentary:
            cluster = 0
        elif a in active:
            cluster = 1
        elif a in excersize:
            cluster = 2

        predictions[i, 0] = int(a)
        predictions[i, 1] = int(p)
        predictions[i, 2] = int(s)
        predictions[i, 3] = int(cluster)

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

    return centroids


def assign_clusters(data, centroids, distance_method='cossim'):

    if distance_method == 'eclud':
        result = ecludian_distance(data, centroids)
        assignments = np.argmin(result, axis=0)
    elif distance_method == 'cossim':
        result = cos_sim(data, centroids)
        assignments = np.argmax(result, axis=0)
    else:
        print("ERROR: Invalid distance method", distance_method)
        sys.exit(1)

    return assignments

def ecludian_distance(data, centroids):
    final_result = np.empty((centroids.shape[0], data.shape[0]))
    for cent in range(0, centroids.shape[0]):
        subtracted = data - centroids[cent, :]
        normalized = np.linalg.norm(subtracted, axis=1)
        final_result[cent, :] = normalized

    return final_result


def cos_sim(data, centroids):
    final_result = np.empty((centroids.shape[0], data.shape[0]))
    for cent in range(0, centroids.shape[0]):
        c = centroids[cent, :]
        result = cossim_single(data, c)

        final_result[cent, :] = result

    #final_result = cosine_similarity(data, centroids).T

    return final_result


def cossim_single(data, c):
    top = np.sum(data * c, axis=1)
    bottom_x = np.linalg.norm(data, axis=1)
    bottom_c = np.linalg.norm(c)
    result = top / (bottom_x * bottom_c)

    return result

def display_results(data, centroids, assignments):
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='rainbow')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')
    plt.show()


def still_changes(old_centroids, current_centroids, attempts):
    subtracted = np.sum(np.absolute(np.absolute(old_centroids) - np.absolute(current_centroids)))
    #print(attempts, np.sum(np.absolute(old_centroids)), np.sum(np.absolute(current_centroids)), subtracted)

    #print(subtracted)
    if np.array_equal(old_centroids, current_centroids) or attempts > 2000 or subtracted < 0.1:
        return False
    else:
        return True


def min_coherence(data, centroids, assignments):
    dist = 0
    k = centroids.shape[0]
    for cent in range(0, k):

        mask = (assignments == cent)
        cluster = data[mask]
        subtracted_cluster = cluster - centroids[cent, :]
        dist += np.sum(np.linalg.norm(subtracted_cluster, axis=1))

    dist = dist / k

    return dist


def max_separation(centroids):

    k = centroids.shape[0]
    dist = 0
    for cent in range(0, k):
        # remove centroids
        compare = centroids[cent, :]
        remaining = np.delete(centroids, cent, axis=0)
        subtracted = remaining - compare
        dist += np.sum(np.linalg.norm(subtracted, axis=1))

    dist = dist / k

    return dist


def mean_entropy(data, centroids, assignments, prediction):
    total_points = data.shape[0]
    result = []
    for cent in range(0, centroids.shape[0]):
        mask = (assignments == cent)
        predict = prediction[mask]

        # Get the counts in each category in each
        unique, counts = np.unique(predict, return_counts=True)
        c_total = predict.shape[0]
        result.append(np.sum((c_total / total_points) * -1 * (counts / c_total) * np.log2(counts / c_total)))

    return np.sum(result)

def calc_cluster_entropy(counts):
    return -()


def run_clustering(data, prediction, k=2):
    # create n centroides by picking random values in each feature range (column)
    centroids = np.random.uniform(0, 1, (k, data.shape[1]))
    centroids_old = np.random.uniform(0, 0.5, (k, data.shape[1]))
    print(centroids.shape)

    assignments = []
    attempts = 0
    total_attempts = 0
    best_centroids = np.copy(centroids)
    best_assignments = []
    best_coherence = 100000000
    coherence = best_coherence - 1
    tries = 0
    best_seperation = 0
    max_tries = 5
    total_assign = 0
    total_cent = 0
    while tries < max_tries:
        while still_changes(centroids_old, centroids, attempts):
            centroids_old = np.copy(centroids)

            assign_start = time.time()
            assignments = assign_clusters(data, centroids)
            total_assign += time.time() - assign_start

            cent_start = time.time()
            centroids = adjust_cluster_center(data, centroids, assignments)
            total_cent += time.time() - cent_start

            attempts += 1

            #display_results(data, centroids, assignments)

        coherence = min_coherence(data, centroids, assignments)
        seperation = max_separation(centroids)

        entropy = mean_entropy(data, centroids, assignments, prediction)
        print('Entropy', entropy)

        if coherence < best_coherence and seperation > best_seperation:
            best_centroids = np.copy(centroids)
            best_assignments = np.copy(assignments)
            best_coherence = coherence
            best_seperation = seperation
            tries = 0
        else:
            tries += 1

        # Randomly move 2 centroids
        moves = int(centroids.shape[0] / 3)
        print(moves)
        centroids_to_move = np.random.randint(0, centroids.shape[0], moves)
        for move in range(0, moves):
            value = np.random.uniform(0, 1, (1, data.shape[1]))
            centroids[centroids_to_move[move], :] = value

        print("Attempts", attempts, ' tries left', max_tries - tries, ' assign time', total_assign, ' cent time', total_cent)
        total_attempts += attempts
        attempts = 0

    print("Attempts", total_attempts)
    if data.shape[1] == 2:  # Graph 2D arrays
        display_results(data, best_centroids, best_assignments)

    entropy = mean_entropy(data, best_centroids, best_assignments, prediction)
    print('Entropy', entropy)




d = np.array([[5, 10]])
c = np.array([1, 4])

value = 1 - cossim_single(d, c)
print(value)
value = spatial.distance.cosine(d, c)
print(value)


#loaded_data = load_data(filename='./SyntheticData.txt', type='txt')
loaded_data = load_data(filename='./pca.npy', type='npy')



#print(result)
#print(result.shape)

print("cossim")
predictions = create_predictions(loaded_data.shape)
all_activities = predictions[:, 0]
three_cluster = predictions[:, 3]
run_clustering(loaded_data, all_activities, 19)


