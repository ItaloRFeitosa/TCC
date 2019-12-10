import os
import sys
import json
from features_extraction import xception
from features_extraction import vgg16
from features_extraction import resnet50
from features_extraction import inceptionv3
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
                        ],
        'media' : [0, 0, 0, 0, 0, 0, 0]
    }
    predictions['preds'] = []

    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'rbf', False), 'Xception + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'rbf', True), 'Xception + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'linear', False), 'Xception + SVM (Kernel linear)'))
    predictions['preds'].append(make_dict(xception.svm(imgpath, modelspath, 'linear', True), 'Xception + normalização + SVM (Kernel linear)'))

    predictions['preds'].append(make_dict(vgg16.svm(imgpath, modelspath, 'rbf', False), 'VGG16 + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(vgg16.svm(imgpath, modelspath, 'rbf', True), 'VGG16 + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(vgg16.svm(imgpath, modelspath, 'linear', False), 'VGG16 + SVM (Kernel linear)'))
    predictions['preds'].append(make_dict(vgg16.svm(imgpath, modelspath, 'linear', True), 'VGG16 + normalização + SVM (Kernel linear)'))

    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'rbf', False), 'ResNet50 + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'rbf', True), 'ResNet50 + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'linear', False), 'ResNet50 + SVM (Kernel linear)'))
    predictions['preds'].append(make_dict(resnet50.svm(imgpath, modelspath, 'linear', True), 'ResNet50 + normalização + SVM (Kernel linear)'))

    predictions['preds'].append(make_dict(inceptionv3.svm(imgpath, modelspath, 'rbf', False), 'InceptionV3 + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(inceptionv3.svm(imgpath, modelspath, 'rbf', True), 'InceptionV3 + normalização + SVM (Kernel Rbf)'))
    predictions['preds'].append(make_dict(inceptionv3.svm(imgpath, modelspath, 'linear', False), 'InceptionV3 + SVM (Kernel linear)'))
    predictions['preds'].append(make_dict(inceptionv3.svm(imgpath, modelspath, 'linear', True), 'InceptionV3 + normalização + SVM (Kernel linear)'))

    for preds in predictions['preds']:
        for i, pred in enumerate(preds['pred']):
            predictions['media'][i] += pred

    predictions['media'] = list([ m/16 for m in predictions['media']])
    return predictions

def main(arg1):
    predictions = predict(arg1)
    filename = arg1.rsplit('.', 1)[0] + '.json'
    path = os.getcwd() + '\\classifiers\\predictions\\'+filename
    with open(path, 'w') as json_file:
        json.dump(predictions, json_file)
    

if __name__ == "__main__":
    main(sys.argv[1])