import os
import sys
import json
from features_extraction import vgg16
from features_extraction import resnet50
from features_extraction import xception
import numpy as np


def make_dict(pred, name):
        return {
            'name' : name,
            'pred' : list(pred),  
            'max' : int(np.argmax(pred))
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
                        'Melanocytic nevi',
                        'Vascular'
                        ]
    }
    predictions['preds'] = []
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'rbf', False), 'Xception + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'rbf', True), 'Xception + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'linear', True), 'Xception + normalização + SVM (Kernel linear)'))
    predictions['preds'].append(make_dict(vgg16.svm(imgpath, modelspath, 'rbf', True), 'VGG16 + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(vgg16.svm(imgpath, modelspath, 'linear', True), 'VGG16 + normalização + SVM (Kernel linear)'))
    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'rbf', True), 'ResNet50 + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'linear', True), 'ResNet50 + normalização + SVM (Kernel linear)'))

    return predictions

def main(arg1):
    predictions = predict(arg1)
    filename = arg1.rsplit('.', 1)[0] + '.json'
    path = os.getcwd() + '\\classifiers\\predictions\\'+filename
    with open(path, 'w') as json_file:
        json.dump(predictions, json_file)
    

if __name__ == "__main__":
    main(sys.argv[1])