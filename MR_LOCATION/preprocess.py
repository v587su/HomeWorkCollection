import pandas as pd
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

from util import *


def main(_data, _station):
    # for row in _data.iterrows():
    #     matrix = get_matrix(row, _station)
    get_trail(_data)


if __name__ == '__main__':
    crt_path = path.dirname(path.abspath(__file__))
    data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
    station_path = path.join(crt_path, 'data_file', 'gongcan.csv')
    data = pd.read_csv(data_path)
    station = pd.read_csv(station_path, index_col=['RNCID', 'CellID'])
    main(data, station)
