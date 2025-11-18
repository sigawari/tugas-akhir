import numpy as np

DATA_PATH = 'C:\Users\hp\multimedia\ta-code\lstm\MP_Data\halo\sequence_1.npy'

data = np.load(DATA_PATH)
print(f"{data.shape()}")