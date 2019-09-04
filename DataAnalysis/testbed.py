import numpy as np

M = np.random.randn(100, 100) 

row_means = np.mean(M, axis=1)

for row in M: 
    row_mean = np.mean(row)
    row_variance = np.var(row)

print(row_means)