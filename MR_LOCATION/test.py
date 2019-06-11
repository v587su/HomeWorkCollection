from util import *
import os.path as path
import pandas as pd

crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'matrix.csv')
data = pd.read_csv(data_path)
print(data.shape)