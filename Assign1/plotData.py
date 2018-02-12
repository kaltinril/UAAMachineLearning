import pandas as pd
import numpy as np


class PlotData:
    def __init__(self):
        self.data = None
        self.rows = None
        self.cols = None
        self.X = None
        self.Y = None
        self.xMin = 0.0
        self.xMax = 0.0
        self.yMin = 0.0
        self.yMax = 0.0
        self.headers = ''

    def load_data(self, filename):
        self.data = pd.read_csv(filename)
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]

        self.headers = self.data.dtypes.index

        self.data = self.data.values
        self.data = self.data[np.arange(0, self.rows), :]

        self.set_data_points()
        self.store_original_min_max()
        self.normalize_data_points()

    def set_data_points(self):
        # Force the arrays to be floats, otherwise we get all 0 or 1 during normalization
        self.X = np.array(self.data[:, 1], dtype=float)
        self.Y = np.array(self.data[:, 2], dtype=float)

    def store_original_min_max(self):
        # Force the values to be floats, so we don't get all 0 and 1
        self.xMin = float(min(self.X))
        self.xMax = float(max(self.X))
        self.yMin = float(min(self.Y))
        self.yMax = float(max(self.Y))

    def normalize_data_points(self):
        self.X = (self.X - self.xMin) / (self.xMax - self.xMin)
        self.Y = (self.Y - self.yMin) / (self.yMax - self.yMin)

    # Allows us to get the original data points back
    def denormalize_data_points(self):
        self.X = (self.X * (self.xMax - self.xMin)) + self.xMin
        self.Y = (self.Y * (self.yMax - self.yMin)) + self.yMin

    # Denormalize a single point and return it back
    def denormalize_single_point(self, x, y):
        xd = (x * (self.xMax - self.xMin)) + self.xMin
        yd = (y * (self.yMax - self.yMin)) + self.yMin
        return xd, yd
