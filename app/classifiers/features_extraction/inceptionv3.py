from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(path):
    tensor = path_to_tensor(path)
    inceptionv3_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    return inceptionv3_model.predict(tensor).flatten()

