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
from TEST3 import SEModule,AdaptiveReweight,CE,OPPORTUNITY_CE,ReparamLargeKernelConv,ELK,ELK_CNN,ChannelAttention,SpatialAttention,resnet
import sys
arg = sys.argv
name = str(arg[1])
modelfile = str(arg[2])
database = str(arg[3])
fold_path = str(arg[4])
datapath = '/home/pi/test/'+database+'/'


X = torch.from_numpy(np.load(datapath+'x_test.npy')[0:100]).float()
y = torch.from_numpy(np.load(datapath+'y_test.npy')[0:100]).long()



model = torch.load(modelfile,map_location=torch.device('cpu'))
model.eval()  # 设置模型为评估模式
with torch.no_grad():     
    single_start_time = time.time()
    single_prediction = model(X[0].unsqueeze(0)).argmax(dim=0).cpu()
    single_end_time = time.time()   
    start_time = time.time()    
    memory_usage = psutil.virtual_memory().used
    # 这里我们直接使用全部数据进行评估，实际中你可能需要一个独立的测试集
    predictions = model(X.to(device)).argmax(dim=1).cpu()
    end_time = time.time()

    single_start_time1 = time.time()
    single_prediction = model(X[0].unsqueeze(0)).argmax(dim=0).cpu()
    single_end_time1 = time.time()
    # memory_usage = psutil.virtual_memory().used

    
    f1 = f1_score(y, predictions, average='weighted')
    acc = accuracy_score(y, predictions)


# 输出 F1 分数和准确率
print(f'F1 Score: {f1:.4f}') 
print(f'Accuracy: {acc:.4f}')
#print(f'Single Data Point Run Time: {(single_end_time - single_start_time)*1000:.6f} milliseconds')
print(f'Single Data Point Run Time: {(single_end_time1 - single_start_time1)*1000:.6f} milliseconds')
# 输出运行时间
print(f'Run Time: {end_time - start_time:.2f} seconds')


# 输出模型参数数量
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')


# 获取内存使用信息
mem = psutil.virtual_memory()

# 打印内存总量
print(f"Total: {mem.total / 1024 ** 2} MB")

# 打印可用内存
print(f"Available: {mem.available / 1024 ** 2} MB")

# 打印使用中的内存
print(f"Used: {(mem.total - mem.available) / 1024 ** 2} MB")

# 打印内存使用百分比
print(f"Percentage: {mem.percent}%")










import os
import time
import logging

# Return CPU temperature as a character string                                     
def  getCPUtemperature():
     res  = os.popen( 'vcgencmd measure_temp' ).readline()
     return (res.replace( "temp=" ," ").replace(" 'C\t "," "))

# Return RAM information (unit=kb) in a list                                      
# Index 0: total RAM                                                              
# Index 1: used RAM                                                                
# Index 2: free RAM                                                                
def  getRAMinfo():
     p  = os.popen( 'free' )
     i  = 0
     while  1 :
         i  = i  + 1
         line  = p.readline()
         if  i == 2 :
             return (line.split()[ 1 : 4 ])
 
# Return % of CPU used by user as a character string                               
def  getCPUuse():
     return ( str (os.popen( "top -n1 | awk '/Cpu\(s\):/ {print $2}'" ).readline().strip()))
 
# Return information about disk space as a list (unit included)                    
# Index 0: total disk space                                                        
# Index 1: used disk space                                                        
# Index 2: remaining disk space                                                    
# Index 3: percentage of disk used                                                 
def  getDiskSpace():
     p = os.popen( "df -h /" )
     i = 0
     while 1 :
         i  = i  + 1
         line  = p.readline()
         if i == 2 :
             return (line.split()[ 1 : 5 ])

def get_info():
     
     # CPU informatiom
     CPU_temp  = getCPUtemperature()
     CPU_usage  = getCPUuse()
     
     # RAM information
     # Output is in kb, here I convert it in Mb for readability
     RAM_stats  = getRAMinfo()
     RAM_total  = round ( int (RAM_stats[ 0 ])  / 1024 , 1 )
     RAM_used  = round ( int (RAM_stats[ 1 ])  / 1024 , 1 )
     RAM_free = round(int (RAM_stats[2]) / 1000,1)
     
     #DISK
     DISK_stats= getDiskSpace()
     DISK_total = DISK_stats[0]
     DISK_used = DISK_stats[1]
     DISK_prec = DISK_stats[3]
     
     
     
     print('CPU Tempertaure = '+CPU_temp)
     print('CPU Use ='+CPU_usage)
     
     print('RAM Total = '+str(RAM_total)+'MB')
     print('RAM Used = '+str(RAM_used)+'MB')
     print('RAM Free = '+str(RAM_free)+'MB')
     
     print('DISK Total Space = '+str(DISK_total)+'B')
     print('DISK Used Space = '+str(DISK_used)+'B')
     print('DISK Used Percentage = '+str(DISK_prec))
     
get_info()





#save as log.txt
with open('/home/pi/new/log4.txt', 'a') as f:
    f.write(f'Model Name: {name}\n')
    f.write(f'Dataset name: {database}\n')
    #f.write(f'F1 Score: {f1:.4f}\n')
    #f.write(f'Accuracy: {acc:.4f}\n')
    f.write(f'Run Time: {end_time - start_time:.6f}\n')
    f.write(f'Single Data Point Run Time: {(single_end_time1 - single_start_time1)*1000:.2f}\n')
    f.write(f'Average Memory Usage: {memory_usage / 1024**2:.2f}\n')
    f.write(f'Number of parameters: {num_params}\n')
    f.write('\n')  #add a line to seperate
