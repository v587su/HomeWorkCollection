from util import *
import os.path as path
import pandas as pd
import matplotlib.pyplot as plt

crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'total_loss.csv')
data = pd.read_csv(data_path, header=None).values.tolist()
plt.plot(range(0,len(data),20), data[::20])
plt.show()
