import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import time
import psutil
import numpy as np
# 检查是否有可用的 GPU，如果有则使用 GPU

import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
arg = sys.argv
name = str(arg[1])
modelfile = str(arg[2])
database = str(arg[3])
fold_path = str(arg[4])
# name = 'resnext50_wisdm_model.pt'
# modelfile = '/home/pi/HAR1/MSAP_WISDM/resnext50_wisdm_model.pt'
# database = 'WISDM'
# fold_path = '/home/pi/HAR1/HAR/MSAP_WISDM'+'/'
datapath = '/home/pi/test/'+database+'/'

from TEST3 import SEModule,AdaptiveReweight,CE,UCI_CE,ReparamLargeKernelConv,ELK,ELK_CNN,ChannelAttention,SpatialAttention,resnet
X = torch.from_numpy(np.load(datapath+'x_test.npy')).float()
y = torch.from_numpy(np.load(datapath+'y_test.npy')).long()


model = torch.load(modelfile, map_location=torch.device('cpu'))
model.eval()  # 设置模型为评估模式
with torch.no_grad():     
    single_start_time = time.time()
    single_prediction = model(X[0].unsqueeze(0)).argmax(dim=0).cpu()
    single_end_time = time.time()   
    time_list = []
    memory_usage = psutil.virtual_memory().used
    for i in range(100):
        single_start_time1 = time.time()
        single_prediction = model(X[i].unsqueeze(0)).argmax(dim=0).cpu()
        single_end_time1 = time.time()
        time_list.append((single_end_time1 - single_start_time1)*1000)

# 读取现有的CSV文件
df = pd.read_csv('/home/pi/new/data.csv')

# 将time_list作为新的列添加到数据框中
df[name] = pd.Series(time_list)

# 将数据框写回CSV文件
df.to_csv('/home/pi/new/data.csv', index=False)



