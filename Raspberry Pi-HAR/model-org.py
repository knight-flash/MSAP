import os
import torch
import time
import psutil
import subprocess
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def evaluate_models(folder_path,database):
    # 获取文件夹内所有.pt文件
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    # 为每个模型文件进行评估
    for model_file in model_files:
        # 加载模型
        
        modelname = model_file
        modelfile = os.path.join(folder_path, model_file)
        print(modelname,modelfile,database,folder_path)
        #subprocess.call(["python", "model-eval.py",modelname,modelfile,database,folder_path])
        print(folder_path+'/'+"model-eval.py")
        subprocess.call(["python", folder_path+'/'+"model-eval.py",modelname,modelfile,database,folder_path])

        time.sleep(25)

for database in ['PAMAP2']:
# for database in ['WISDM']:
    filepath = '/home/pi/new/'+database
    evaluate_models(filepath,database)
