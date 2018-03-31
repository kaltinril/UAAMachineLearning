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




pca = np.load('./pca.npy')

predictions = create_predictions(pca.shape)
print(predictions)
print(predictions.shape)
print(predictions[(24*60*63)])

# Pick some Centroids
a = pca[:, 3:4]
b = pca[:, 7:8]
c = pca[:, 9:10]

# distance between two points in 1D
d = pca-a
e = pca-b
f = pca-c

# calculate the square a^2 for each 1D distance
d = d * d
e = e * e
f = f * f

# a^2 + b^2 + c^2 (add all 1D squared distances)
d = np.sum(d, axis=1, keepdims=True)
e = np.sum(e, axis=1, keepdims=True)
f = np.sum(f, axis=1, keepdims=True)

# join the values
h = np.hstack((d,e))
h = np.hstack((h,f))

print(h.shape)

# Find the max argument for each row
i = np.argmax(h, axis=1)

print(i.shape)
print(d.shape)
print(h[1:2,:])
print(i[1:2])
print(h[i[1:2]])
print(h[2])
print(h)

j = h[i]

print(j.shape)
print(i)
print(i.max())
print(i.min())