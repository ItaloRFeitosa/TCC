
import os
from classifiers.features_extraction import inceptionv3
from classifiers.features_extraction import resnet50
from classifiers.features_extraction import xception
import numpy as np

def make_dict(pred, name):
        return {
            'name' : name,
            'pred' : pred,  
            'max' : np.argmax(pred)
        }

def predict(filename):
    modelspath = os.getcwd() + '\\classifiers\\models'

    imgpath = os.getcwd() + '\\static\\upload\\' +filename
    
    print(imgpath)
    predictions = { 'preds': [],
        'labels' : ['akiec','bcc','bkl','df','mel','nv','vasc'],
        'descricao' : ['Actinic Keratoses',
                        'Basal cell carcinoma',
                        'Benign keratosis',
                        'Dermatofibroma',
                        'Melanoma',
                        'Melanocytic nev',
                        'Vascular'
                        ]
    }
    predictions['preds'] = []
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'rbf', False), 'Xception + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'rbf', True), 'Xception + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'rbf', True), 'ResNet50 + normalização + SVM (Kernel Rbf)'))
    
    return predictions