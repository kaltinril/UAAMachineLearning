import numpy as np
import matplotlib.pyplot as plt

final_array = np.load('./data_combined.npy')

col = 3

result = np.fft.fft(final_array[0:126, col:col+1].flatten())

plt.plot(final_array[0:126, col:col+1], 'v-')
plt.show()

plt.plot(np.real(result), 'o-')
plt.show()

plt.plot(np.imag(result), 'x-')
plt.show()

fnames = None
arrays = None
final_array = None
