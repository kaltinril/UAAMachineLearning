import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1, 3], [-1, -2, -3], [1, -3, 2], [1, 1, -2], [-1, 2, 2], [1, 2, 1]])
pca = PCA(.80)  #n_components=X.shape[1])
pca.fit(X)
X_pca = pca.transform(X)
X_pca_reversed = pca.inverse_transform(X_pca)

print(X_pca)
print(X)
print(X_pca_reversed)
print(X - X_pca_reversed)
print(np.average(X - X_pca_reversed, axis=0))
print("hhhhhhh")

print(pca.components_)
print("dddddddddd")

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)



def stuff():
    # Sort X by the explained_variance_ratio
    Y = pca.explained_variance_ratio_

    Y = np.array(Y)
    print("Y",Y)
    print("X[0]",X[0])
    indexes = Y.argsort()  # Return the index positions of the variance in ASCENDING order
    indexes = np.flip(indexes, axis=0)  # Flip to convert to DESCENDING order of index positions
    print("ind",indexes)
    sorted_x = X.copy()

    for i in range(len(sorted_x)):
        sorted_x[i] = X[i][indexes]  # Use those indexes.

    print("Sorted",sorted_x)
    print("X", X)


    print("singlevalues",pca.singular_values_)
