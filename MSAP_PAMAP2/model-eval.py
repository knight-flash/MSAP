import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import time
import psutil
import numpy as np
# 检查是否有可用的 GPU，如果有则使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
arg = sys.argv
# name = str(arg[1])
# modelfile = str(arg[2])
# database = str(arg[3])
# fold_path = str(arg[4])+'\\'
name = 'ATT_pamap2_model.pt'
modelfile = 'F:\lunwen\树莓派\HAR\MSAP_PAMAP2\ATT_pamap2_model.pt'
database = 'PAMAP2'
fold_path = 'F:\lunwen\树莓派\HAR\MSAP_PAMAP2'+'\\'
datapath = 'F:\\lunwen\\HAR\\dataset\\'+database+'\\MSC\\'
print(fold_path+'MSAP.py')
sys.path.append(fold_path+'MSAP')
sys.path.append(fold_path+'TEST1.py')
sys.path.append(fold_path+'TEST2.py')
import GatedFCN,GatedRes2NetBottleneck,EfficientChannelAttention,Shrinkage_SE,Shrinkage_ECA,ECA
import GatedFCN2,GatedRes2NetBottleneck1,GatedFCN3,GatedRes2NetBottleneck2,SEModule
import ResNet,Bottleneck,LSTM_CNN,CNN,CNN_GRU,ATT,LSTM
X = torch.from_numpy(np.load(datapath+'x_test.npy')).float()
y = torch.from_numpy(np.load(datapath+'y_test.npy')).long()
with open('log2.txt', 'a') as f:
    f.write(f'Model Name: {name}\n')


model = torch.load(modelfile)

# 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    start_time = time.time()
    # 这里我们直接使用全部数据进行评估，实际中你可能需要一个独立的测试集
    predictions = model(X.to(device)).argmax(dim=1).cpu()
    end_time = time.time()
    memory_usage = psutil.virtual_memory().used
    f1 = f1_score(y, predictions, average='weighted')
    acc = accuracy_score(y, predictions)

# 输出 F1 分数和准确率
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy: {acc:.4f}')

# 输出运行时间
print(f'Run Time: {end_time - start_time:.2f} seconds')

# 输出模型参数数量
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')

#save as log.txt
with open('log2.txt', 'a') as f:
    f.write(f'Model Name: {name}\n')
    f.write(f'Dataset name: {database}\n')
    f.write(f'F1 Score: {f1:.4f}\n')
    f.write(f'Accuracy: {acc:.4f}\n')
    f.write(f'Run Time: {end_time - start_time:.2f}\n')
    f.write(f'Average Memory Usage: {memory_usage / 1024**2:.2f}\n')

    f.write(f'Number of parameters: {num_params}\n')
    f.write('\n')  #add a line to seperate