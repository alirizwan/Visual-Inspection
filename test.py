import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2 as cv

from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders, get_inference_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize, predict
from utils.constants import NEG_CLASS

data_folder = "data1/"
subset_name = "doors"
data_folder = os.path.join(data_folder, subset_name)

batch_size = 13
target_train_accuracy = 0.98
lr = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7
n_cv_folds = 5

data = get_inference_loaders(
    root=data_folder, batch_size=1
)

model_path = f"weights/{subset_name}_model.h5"
model = torch.load(model_path, map_location=device)

predict_localize(model, data, device, thres=heatmap_thres, n_samples=12, show_heatmap=True)

def it_predict():
    input_folder = '/Users/alirizwan/Downloads/Preprocessor/output/NG'
    nok_folder = './results/nok/'
    ok_folder = './results/ok/'

    input_files = []

    for file_path in os.listdir(input_folder):
        if os.path.isfile(os.path.join(input_folder, file_path)):
            input_files.append(file_path)

    for image in input_files:

        if(image == '.DS_Store'): continue

        input_file_name = os.path.join(input_folder, image)

        print('processing', input_file_name)

        img = Image.open(input_file_name)

        prob, class_pred, heatmap, result = predict(model, img, device, thres=heatmap_thres)

        if (class_pred == 0):
            cv.imwrite(ok_folder + image, result)
        else:
            cv.imwrite(nok_folder + image, result)

#it_predict()