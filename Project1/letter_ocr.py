import numpy as np
import pandas as pd
import ann_simple

data = pd.read_csv('./Letters.csv', header=None)

rows = data.shape[0]
cols = data.shape[1]
data = data.values

X = data[:, (range(1, cols))]
Y = data[:, 0] # Snag the first column corresponding to the letter

nn = ann_simple.ANN()
result = nn.foward_learning(X[0]) # 1 row at a time

print(result)


