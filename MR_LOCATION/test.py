from util import *
import os.path as path
import pandas as pd

crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
data = pd.read_csv(data_path)
print(get_trail(data))
print([len(a) for a in get_trail(data)])