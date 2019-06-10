import pandas as pd
import numpy as np
import torch.cuda

TRAIL_INTERVAL = 500
TRAIL_LENGTH = 6
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
BATCH_SIZE = 32
EPOCHS = 200


def clean_matrix(matrix):
    last_available = None
    for row in matrix.iterrows():
        if pd.isnull(row[1]['Latitude']):
            matrix.loc[row[0]] = last_available
        else:
            last_available = row[1]
    return matrix


def get_matrix(row, station):
    matrix = np.zeros((6, 5))
    feature_list = ['RNCID', 'CellID', 'Dbm', 'AsuLevel', 'SignalLevel']
    for i in range(1, 7):
        for j in range(len(feature_list)):
            column_name = '{}_{}'.format(feature_list[j], str(i))
            matrix[i - 1, j] = row[1][column_name]
    matrix = pd.DataFrame(matrix, columns=feature_list).join(station,
                                                             on=['RNCID',
                                                                 'CellID'])
    matrix = clean_matrix(matrix)
    return matrix


def get_trail(data):
    trails = []
    for i in data.groupby('IMSI'):
        last_time = 0
        trail_temp = []
        new_token = True
        for row in i[1].iterrows():
            if -TRAIL_INTERVAL > last_time - row[1]['MRTime']:
                if not new_token:
                    trails.append(trail_temp)
                new_token = False
                trail_temp = []
            trail_temp.append(str(row[0]))
            last_time = row[1]['MRTime']
    trails = split_trail(trails)
    return trails


def split_trail(trails):
    trails_splited = []
    for i, trail in enumerate(trails):
        while len(trail) > TRAIL_LENGTH:
            trails_splited.append(trail[:TRAIL_LENGTH])
            trail = trail[TRAIL_LENGTH:]
        if len(trail) < TRAIL_LENGTH:
            trail = padding_trail(trail, trails_splited[-1])
        trails_splited.append(trail)
    return trails_splited


def padding_trail(trail, last_trail):
    padding_num = TRAIL_LENGTH - len(trail)
    trail = last_trail[-padding_num:] + trail
    return trail
