import subprocess
import sys
import os
import json

def predict(filename):
    python_path = os.path.join(sys.base_exec_prefix,'python.exe')
    command = python_path + ' classifiers/classifiers.py ' + filename
    print(command)
    code = subprocess.call(command, shell=True)
    if(code == 0):
        
        name = filename.rsplit('.', 1)[0] + '.json'
        
        path = os.getcwd() + '\\classifiers\\predictions\\'+name
        with open(path) as json_file:
            preds = json.load(json_file)
            return preds
    # print(code)