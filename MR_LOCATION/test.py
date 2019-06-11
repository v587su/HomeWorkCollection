from util import *
import os.path as path
import pandas as pd

crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'result.csv')
data = pd.read_csv(data_path,header=None)
data.columns = ['pre_lon', 'pre_lat', 'tgt_lon', 'tgt_lat']
data[['pre_lon','tgt_lon']] = data[['pre_lon','tgt_lon']].mul(MAX_LON-MIN_LON).add(MIN_LON)
data[['pre_lat','tgt_lat']] = data[['pre_lat','tgt_lat']].mul(MAX_LAT-MIN_LAT).add(MIN_LAT)
print(data)
