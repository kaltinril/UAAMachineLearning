import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal


def data2np(graph_data, format):
    # Generate a figure with matplotlib
    fig = plt.figure()
    plot = fig.add_subplot(111)

    fig.set_size_inches(9, 5)

    # Plot the data and draw the canvas
    plot.plot(graph_data, format)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.clf()
    plt.close(fig)


    return data

final_array = np.load('./data_combined.npy')

final_fft_array = []

x = np.arange(0, 100, 0.1)
y = np.sin(x)
#result = np.fft.fft(y)
result = np.fft.fft(final_array[0:125, 0:1].flatten())
#real_result = np.real(result)
real_result = np.real(result)
second_result = signal.periodogram(real_result[1:125], 25)

cv2.imshow('orig', data2np(final_array[0:125, 0:1], 'r-'))
cv2.imshow('fft', data2np(real_result[1:125], 'b-'))
cv2.imshow('periodgram', data2np(second_result, 'g-'))
cv2.waitKey()

f, Pxx_den = signal.periodogram(real_result[1:125], 25)
plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()




for sample_start in range(0, final_array.shape[0], 125):
    fft_array = []
    for col in range(0, final_array.shape[1]):

        result = np.fft.fft(final_array[sample_start:sample_start+125, col:col+1].flatten())
        fft_array.append(np.real(result[1:125]).T)  # Strip off the first 

        #orig = data2np(final_array[0:126, col:col+1], 'r-')
        #real = data2np(np.real(result[1:126]), 'b-')
        #imag = data2np(np.imag(result), 'g-')

        #orig = np.vstack((orig, real))

        #cv2.imshow('img', orig)
        #cv2.waitKey()


        # plt.plot(final_array[0:126, col:col+1], 'v-')
        # plt.show()
        #
        # plt.plot(np.real(result), 'o-')
        # plt.show()
        #
        # plt.plot(np.imag(result), 'x-')
        # plt.show()

    # Combine the result back together
    final_fft_array.append(np.vstack(fft_array).T)


print(final_array.shape)
final_array = None

print(len(final_fft_array))

final_fft_array = np.concatenate(final_fft_array)
print(final_fft_array.shape)


final_fft_array = None

