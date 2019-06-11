import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_0 = pd.read_csv('acc_0.csv',header=None)
data_1 = pd.read_csv('acc_1.csv',header=None)
data_2 = pd.read_csv('acc_2.csv',header=None)
data_3 = pd.read_csv('acc_3.csv',header=None)

plt.plot(range(30), data_0.values.tolist(),label='my_loss')
plt.plot(range(30), data_1.values.tolist(), label='ce_loss')
plt.plot(range(30), data_2.values.tolist(), label='mse_loss')
plt.plot(range(30), data_3.values.tolist(), label='average_ce_mse')
plt.legend()
plt.title('Val_Acc of different loss type')
plt.savefig('training_process.png')
