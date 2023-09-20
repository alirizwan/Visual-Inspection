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

# predict_localize(model, data, device, thres=heatmap_thres, n_samples=8, show_heatmap=True)

img = Image.open('./data1/doors/test/defect/230629_132116_0000038921_CAM1_NG.bmp.jpg')

prob, class_pred, heatmap, result = predict(model, img, device, thres=heatmap_thres)