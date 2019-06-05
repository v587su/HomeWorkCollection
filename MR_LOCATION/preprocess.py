import pandas as pd
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from util import *


def main(_data, _station):
    matrixs = []
    for row in _data.iterrows():
        matrix = get_matrix(row, _station)
        matrixs.append(matrix.values.reshape(-1))
    matrixs = np.array(matrixs)
    trail_idx = get_trail(_data)
    return matrixs, trail_idx


if __name__ == '__main__':
    crt_path = path.dirname(path.abspath(__file__))
    data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
    station_path = path.join(crt_path, 'data_file', 'gongcan.csv')
    data = pd.read_csv(data_path)
    station = pd.read_csv(station_path, index_col=['RNCID', 'CellID'])
    mat, trail = main(data, station)
    trail = [','.join(a) + '\n' for a in trail]
    np.savetxt(path.join(crt_path, 'data_file', 'matrix.csv'), mat,
               delimiter=',')
    f = open(path.join(crt_path, 'data_file', 'trail.csv'), 'w')
    f.writelines(trail)
    f.close()
