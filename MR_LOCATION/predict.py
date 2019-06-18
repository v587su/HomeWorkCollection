import torch
import os.path as path
from util import *

crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'test_2g.csv')
trail_path = path.join(crt_path, 'data_file', 'trail_test.csv')
matrix_path = path.join(crt_path, 'data_file', 'matrix_test.csv')

data = pd.read_csv(data_path)
print(data.shape)
trail = np.loadtxt(trail_path, delimiter=',')
print(trail.shape)
matrix = pd.read_csv(matrix_path, header=None)
print(matrix.shape)
# train_idx, val_idx = train_test_split(trail, test_size=0.2, shuffle=True)
test_idx = trail
test_data = DataLoader(Trails(data, test_idx, matrix, train=False),
                       batch_size=BATCH_SIZE)
model = torch.load('model.pkl')
model.eval()
datas = []
for i, (mat, tra, speed) in enumerate(test_data):
    mat = mat.to(device=DEVICE).double()
    speed = speed.to(device=DEVICE).double()
    tra = tra.to(device=DEVICE).double()
    pre = model(mat, speed)
    pres = pre.cpu().detach().numpy().reshape(-1, 2)
    tras = tra.cpu().detach().numpy()
    pres_tras = np.column_stack((tras.reshape(-1, 1), pres))
    datas.append(pres_tras)
result = np.row_stack(datas)
print(result.shape)
result = pd.DataFrame(result, columns=['index', 'Longitude', 'Latitude'])
print(result.sort_values(['index']).drop_duplicates('index'))
result['Longitude'] = result['Longitude'].mul(MAX_LON - MIN_LON).add(MIN_LON)
result['Latitude'] = result['Latitude'].mul(MAX_LAT - MIN_LAT).add(MIN_LAT)
result[['Longitude', 'Latitude']].to_csv('pred.csv', index=False)
