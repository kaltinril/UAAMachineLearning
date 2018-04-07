import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import time


def data2np(graph_data, format):
    # Generate a figure with matplotlib
    fig = plt.figure()
    plot = fig.add_subplot(111)

    fig.set_size_inches(15, 11)

    # Plot the data and draw the canvas
    plot.plot(graph_data, format)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.clf()
    plt.close(fig)

    return data


def perform_pca_reduction(data):
    # run it through PCA
    pca = PCA(0.95, copy=False)  # Only keep first N features that add up to cumulative 95% of the variance
    pca_data = pca.fit_transform(data)

    return pca_data


def run_fft_psd(input_array):
    final_fft_array = []
    for sample_start in range(0, input_array.shape[0], 125):
        # Generate the Fast Fourier Transform
        fft = np.fft.fft(input_array[sample_start:sample_start+125, :], axis=0)
        fft = np.real(fft[1:125, :])  # Strip off the first feature, as it's always weird

        # Calculate the PSD (Power Spectral Density)
        f, pxx_den = signal.periodogram(fft, 25, axis=0)

        # Combine all the rows into a single row
        fft_array = np.reshape(pxx_den, (1, pxx_den.shape[0] * pxx_den.shape[1]))
        final_fft_array.append(fft_array)

        # cv2.imshow('fft', data2np(fft[:, 0:1], 'b-'))
        # cv2.imshow('periodgram', data2np(pxx_den[:, 0:1], 'g-'))
        # cv2.waitKey()

    # Stack each "Sample" on-top of each-other.
    final_fft_array = np.vstack(final_fft_array)
    print('Final Shape', final_fft_array.shape)

    # Save the data off, because we get MEMORY errors
    np.save('./fft_psd1', final_fft_array)

    return final_fft_array


def load_fft_psd():
    final_fft_array = np.load('./fft_psd.npy')
    print(final_fft_array.shape)
    return final_fft_array


def run_pca(fft_pca_data):
    pca = perform_pca_reduction(fft_pca_data)
    print(len(pca))
    print(pca.shape)

    np.save('./pca', pca)


def normalize_data(data):
    print('first', data[0])
    for col in range(0, data.shape[1]):
        column_data = data[:, col:col+1]
        column_data = (column_data - column_data.min()) / (float(column_data.max()) - column_data.min())
        data[:, col:col+1] = column_data

    print('first-after', data[0])

    return data


start = time.time()
input_array = np.load('./data_combined.npy')
input_array = normalize_data(input_array)
data = run_fft_psd(input_array)  # Can't run this and PCA, get Memory Error, have 32 bit python
end = time.time()
print('Total time for FFT and PSD', end-start)

start = time.time()
data = load_fft_psd()
data = normalize_data(data)
run_pca(data)
end = time.time()
print('Total time for PCA', end-start)

pca = np.load('./pca.npy')
cv2.imshow('pca', data2np(pca, '-'))
cv2.waitKey()
