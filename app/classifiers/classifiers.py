
import os
from classifiers.features_extraction import inceptionv3
from classifiers.features_extraction import resnet50
from classifiers.features_extraction import xception



def predict(filename):
    modelspath = os.getcwd() + '\\app\\classifiers\\models'

    imgpath = os.getcwd() + '\\app\\static\\upload\\' +filename
    


    pred_xception_svm_rbf = xception.svm(imgpath, modelspath, 'rbf', False)
    return pred