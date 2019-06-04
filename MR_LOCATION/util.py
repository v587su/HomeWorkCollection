import pandas as pd
import numpy as np


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
            if -20 > last_time - row[1]['MRTime']:
                if not new_token:
                    trails.append(trail_temp)
                new_token = False
                trail_temp = []
            trail_temp.append(row[0])
            last_time = row[1]['MRTime']
    trails = split_trail(trails)
    print([len(a) for a in trails])
    return trails


def split_trail(trails):
    trails_splited = []
    for trail in trails:
        while len(trail) > 30:
            trails_splited.append(trail[:30])
            trail = trail[30:]
        trails_splited.append(trail)
    return trails_splited
