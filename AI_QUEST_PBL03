# 前準備

## Import

!pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet
import numpy as np
import pandas as pd
import codecs
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import torchvision
import os.path as osp 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms
import random
import math
import time
import csv
import cv2

import torch
import torch.nn.functional as F
import glob
import random
import json
%matplotlib inline

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

os.chdir('/content/drive/My Drive/AI QUEST/PBL_03') # カレントディレクトリを指定
os.getcwd()

## Grad Cam

!pip install pytorch-gradcam

# Basic Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# PyTorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import make_grid, save_image

# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

def Grad_Cam(model, train_datasets):
  device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
  model.eval()
  target_layer = model.features
  gradcam = GradCAM(model, target_layer)
  gradcam_pp = GradCAMpp(model, target_layer)
  images = []
  for i in range(10):
      index = random.randint(0, 212)
      first_inputs, _ = train_datasets.__getitem__(index)
      inputs = first_inputs.to(device).unsqueeze(0)
      mask, _ = gradcam(inputs)
      heatmap, result = visualize_cam(mask, first_inputs)

      mask_pp, _ = gradcam_pp(inputs)
      heatmap_pp, result_pp = visualize_cam(mask_pp, first_inputs)

      images.extend([first_inputs.cpu(), heatmap, heatmap_pp, result, result_pp])
  grid_image = make_grid(images, nrow=5)

  return transforms.ToPILImage()(grid_image)

## Open CV

# def binary_threshold(path, back_thresh, fore_thresh, maxValue):
#     img = cv2.imread(path)
#     grayed = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     th, drop_back = cv2.threshold(grayed, back_thresh, maxValue, cv2.THRESH_BINARY)
#     th, clarify_born = cv2.threshold(grayed, fore_thresh, maxValue, cv2.THRESH_BINARY_INV)
    
#     merged = np.minimum(drop_back, clarify_born)
#     return merged

# binary_threshold_for_img = lambda im: binary_threshold(IMAGE_PATH, back_thresh=100, fore_thresh=230, maxValue=255)
    
# def padding_position(x, y, w, h, p):
#     return x - p, y - p, w + p * 2, h + p * 2


# def detect_contour(path, min_size):
#     contoured = cv2.imread(path)
#     forcrop = cv2.imread(path)

#     print("detect" + path)
#     birds = binary_threshold_for_img(path)
#     birds = cv2.bitwise_not(birds)

#     contours, hierarchy = cv2.findContours(birds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     crops = []
#     for c in contours:
#         if cv2.contourArea(c) < min_size:
#             continue

#         x, y, w, h = cv2.boundingRect(c)
#         x, y, w, h = padding_position(x, y, w, h, 80)

#         cropped = forcrop[y:(y + h), x:(x + w)]
#         cropped = cv2.resize(cropped, (2016, 2016))  
#         crops.append(cropped)

#         cv2.drawContours(contoured, c, -1, (0, 0, 255), 3)  
#         cv2.rectangle(contoured, (x, y), (x + w, y + h), (0, 255, 0), 3) 
                
#     return contoured, crops

# def show_contour(path, j):
#     _, crops = detect_contour(path, 20000)
#     for i, c in enumerate(crops):
#         save_path = os.getcwd() + '/' + j + '.jpeg'
#         # cv2.imwrite(save_path, c)

# os.chdir('/content/drive/My Drive/AI QUEST/PBL_03/test') # カレントディレクトリを指定
# for i in range(len(new_test_list)):
#   IMAGE_PATH = new_test_list[i]
#   j = new_test_list[i][45:48]
#   show_contour(IMAGE_PATH, j)

# **Normal**

## Dataset

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'valid': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)), 
                transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ]),
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)), 
                transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),  
                transforms.CenterCrop(resize),  
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)  
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def make_datapath_list(phase="train", target='bridge'):
    if phase == "train":
      rootpath = "/content/drive/My Drive/AI QUEST/PBL_03/"
      target_path = osp.join(rootpath+phase+'/'+target+'/*.jpeg')

      path_list = []  
      for path in glob.glob(target_path):
          path_list.append(path)

      regular_path = osp.join(rootpath+phase+'/'+'regular'+'/*.jpeg')
      for path in glob.glob(regular_path):
          path_list.append(path)

    if phase == "valid":
      rootpath = "/content/drive/My Drive/AI QUEST/PBL_03/"
      target_path = osp.join(rootpath+phase+'/'+target+'*.jpeg')
      path_list = []  
      for path in glob.glob(target_path):
          path_list.append(path)

      regular_path = osp.join(rootpath+phase+'/'+'regular'+'*.jpeg')
      for path in glob.glob(regular_path):
          path_list.append(path)


    elif phase == "test":
      rootpath = "/content/drive/My Drive/AI QUEST/PBL_03/"
      target_path = osp.join(rootpath+phase+'/*.jpeg')
      path_list = []  
      for path in glob.glob(target_path):
          path_list.append(path)

    return path_list
test_list_2 = make_datapath_list(phase="test")

new_test_list_2 = []
for i in range(len(test_list_2)):
  for j in range(len(test_list_2)):
    if i == int(test_list_2[j][45:48]):
      new_test_list_2.append(test_list_2[j])

class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  
        self.transform = transform 
        self.phase = phase  

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path) 
        img_transformed = self.transform(
            img, self.phase) 
        if self.phase == "train":
            label = img_path[46]    
        if self.phase == "valid":
            label = img_path[46]              
        elif self.phase == "test":
            label = np.nan

        if label == "b":
            label = 1
        if label == "h":
            label = 1
        if label == "p":
            label = 1
        if label == "r":
            label = 0

        return img_transformed, label


size = 224
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

test_dataset_2 = HymenopteraDataset(
    file_list=new_test_list_2, transform=ImageTransform(size, mean, std), phase='test')

## Bridge

use_pretrained = True 
bridge_net = models.vgg16(pretrained=use_pretrained)
bridge_net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

bridge_load_path = './best_tuning_bridge.pth'
bridge_load_weights = torch.load(bridge_load_path, map_location={'cuda:0': 'cpu'})
bridge_net.load_state_dict(bridge_load_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bridge_net.to(device)
bridge_net.eval()
bridge_preds = []
for i in range(test_dataset_2.__len__()):
  test_inputs, _ = test_dataset_2.__getitem__(i)
  test_inputs = test_inputs.to(device)
  test_inputs_2 = test_inputs.to('cpu').detach().numpy().copy()
  test_inputs_2 = test_inputs_2[:, ::-1, :]
  test_inputs_2 = torch.from_numpy(test_inputs_2.astype(np.float32))
  test_outputs = bridge_net(test_inputs.unsqueeze(0))
  test_inputs_2 = test_inputs_2.to(device)
  test_outputs_2 = bridge_net(test_inputs_2.unsqueeze(0))
  test_outputs = torch.softmax(test_outputs, dim=1)
  test_outputs_2 = torch.softmax(test_outputs_2, dim=1)
  test_outputs = test_outputs.to('cpu').detach().numpy()
  test_outputs_2 = test_outputs_2.to('cpu').detach().numpy()
  if test_outputs[0][0] + test_outputs_2[0][0]  > 1.1:
    pred = 0
  else:
    pred = 1
  bridge_preds.append(pred)

# Grad_Cam(bridge_net, test_dataset_2)

## Potato

use_pretrained = True  
potato_net = models.vgg16(pretrained=use_pretrained)
potato_net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

potato_load_path = './best_tuning_potota.pth'
potato_load_weights = torch.load(potato_load_path, map_location={'cuda:0': 'cpu'})
potato_net.load_state_dict(potato_load_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
potato_net = potato_net.to(device)
potato_net.eval()
potato_preds = []
for i in range(test_dataset_2.__len__()):
  test_inputs, _ = test_dataset_2.__getitem__(i)
  test_inputs = test_inputs.to(device)
  test_inputs_2 = test_inputs.to('cpu').detach().numpy().copy()
  test_inputs_2 = test_inputs_2[:, ::-1, :]
  test_inputs_2 = torch.from_numpy(test_inputs_2.astype(np.float32))
  test_inputs_2 = test_inputs_2.to(device)
  test_outputs = potato_net(test_inputs.unsqueeze(0))
  test_outputs_2 = potato_net(test_inputs_2.unsqueeze(0))
  test_outputs = torch.softmax(test_outputs, dim=1)
  test_outputs_2 = torch.softmax(test_outputs_2, dim=1)
  test_outputs = test_outputs.to('cpu').detach().numpy()
  test_outputs_2 = test_outputs_2.to('cpu').detach().numpy()
  if test_outputs[0][0] + test_outputs_2[0][0]  > 0.75: 
    pred = 0
  else:
    pred = 1
  potato_preds.append(pred)

# Grad_Cam(potato_net, test_dataset_2)

# **Resized**

## DataSet Resized

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'valid': transforms.Compose([
                transforms.RandomOrder([
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(
                    resize, scale=(0.8,1.0)),]),
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)  
            ]),
            'train': transforms.Compose([
                transforms.RandomOrder([
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
                transforms.RandomHorizontalFlip(),  
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(
                    resize, scale=(0.8,1.0)),]),
                transforms.ToTensor(),  
                transforms.Normalize(mean, std)  
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),  
                transforms.ToTensor(),  
                transforms.Normalize(mean, std)  
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

def make_datapath_list(phase="train", target='bridge'):
    rootpath = "/content/drive/My Drive/AI QUEST/PBL_03/"
    if phase == "train":
      target_path = osp.join(rootpath+phase+'/'+target+'_resized/*.jpeg')
      path_list = [] 
      for path in glob.glob(target_path):
          path_list.append(path)

    if phase == "valid":
      rootpath = "/content/drive/My Drive/AI QUEST/PBL_03/"
      target_path = osp.join(rootpath+phase+'/'+target+'_resized/*.jpeg')
      path_list = [] 
      for path in glob.glob(target_path):
          path_list.append(path)

    elif phase == "test":
      rootpath = "/content/drive/My Drive/AI QUEST/PBL_03/"
      target_path = osp.join(rootpath+phase+'_resized/*.jpeg')
      path_list = []  
      for path in glob.glob(target_path):
          path_list.append(path)

    return path_list

horn_tra_list = make_datapath_list(phase="train", target='horn')
horn_val_list = make_datapath_list(phase="valid", target='horn')

potato_tra_list = make_datapath_list(phase="train", target='potato')
potato_val_list = make_datapath_list(phase="valid", target='potato')

regular_tra_list = make_datapath_list(phase="train", target='regular')
regular_val_list = make_datapath_list(phase="valid", target='regular')

test_list = make_datapath_list(phase="test")

horn_train_list = horn_tra_list + regular_tra_list
potato_train_list = potato_tra_list + regular_tra_list 
horn_valid_list = horn_val_list + regular_val_list
potato_valid_list = potato_val_list + regular_val_list

new_test_list = []
for i in range(len(test_list)):
  for j in range(len(test_list)):
    if i == int(test_list[j][53:56]):
      new_test_list.append(test_list[j])

class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list 
        self.transform = transform  
        self.phase = phase 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)  

        img_transformed = self.transform(
            img, self.phase)  

        if self.phase == "train":
            label = img_path[46:49]
        if self.phase == "valid":
            label = img_path[46:49]           
        elif self.phase == "test":
            label = np.nan

        if label == "bri":
            label = 1
        elif label == "hor":
            label = 1
        elif label == "pot":
            label = 1
        elif label == "reg":
            label = 0

        return img_transformed, label

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

horn_train_dataset = HymenopteraDataset(
    file_list=horn_train_list, transform=ImageTransform(size, mean, std), phase='train')

horn_valid_dataset = HymenopteraDataset(
    file_list=horn_valid_list, transform=ImageTransform(size, mean, std), phase='valid')

potato_train_dataset = HymenopteraDataset(
    file_list=potato_train_list, transform=ImageTransform(size, mean, std), phase='train')

potato_valid_dataset = HymenopteraDataset(
    file_list=potato_valid_list, transform=ImageTransform(size, mean, std), phase='valid')

test_dataset = HymenopteraDataset(
    file_list=new_test_list, transform=ImageTransform(size, mean, std), phase='test')

batch_size = 64

horn_train_dataloader = torch.utils.data.DataLoader(
    horn_train_dataset, batch_size=batch_size, shuffle=True)
horn_val_dataloader = torch.utils.data.DataLoader(
    horn_valid_dataset, batch_size=batch_size, shuffle=True)

potato_train_dataloader = torch.utils.data.DataLoader(
    potato_train_dataset, batch_size=batch_size, shuffle=True)
potato_val_dataloader = torch.utils.data.DataLoader(
    potato_valid_dataset, batch_size=batch_size, shuffle=True)

dataloaders_dict = {"horn_train": horn_train_dataloader, "horn_val": horn_val_dataloader,
                    "potato_train": potato_train_dataloader, "potato_val": potato_val_dataloader}

# **VGG(20°)**

## Resized-Horn

use_pretrained = True  
horn_vgg_net = models.vgg16(pretrained=use_pretrained)
horn_vgg_net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

horn_load_path = './best_tuning_horn_infracted_resized_2.pth'
horn_load_weights = torch.load(horn_load_path, map_location={'cuda:0': 'cpu'})
horn_vgg_net.load_state_dict(horn_load_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
horn_vgg_net = horn_vgg_net.to(device)
horn_vgg_net.eval()
horn_vgg_preds = []
for i in range(test_dataset.__len__()):
  test_inputs, _ = test_dataset.__getitem__(i)
  test_inputs = test_inputs.to(device)
  test_inputs_2 = test_inputs.to('cpu').detach().numpy().copy()
  test_inputs_2 = test_inputs_2[:, ::-1, :]
  test_inputs_2 = torch.from_numpy(test_inputs_2.astype(np.float32))
  test_inputs_2 = test_inputs_2.to(device)
  test_outputs = horn_vgg_net(test_inputs.unsqueeze(0))
  test_outputs_2 = horn_vgg_net(test_inputs_2.unsqueeze(0))
  test_outputs = torch.softmax(test_outputs, dim=1)
  test_outputs_2 = torch.softmax(test_outputs_2, dim=1)
  test_outputs = test_outputs.to('cpu').detach().numpy()
  test_outputs_2 = test_outputs_2.to('cpu').detach().numpy()
  if test_outputs[0][0] + test_outputs_2[0][0]  > 0.5:
    pred = 0
  else:
    pred = 1
  horn_vgg_preds.append(pred)

Grad_Cam(horn_vgg_net, test_dataset)

## Resized-Potato

use_pretrained = True  
potato_vgg_net = models.vgg16(pretrained=use_pretrained)
potato_vgg_net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

potato_load_path = './best_tuning_potato_infracted_resized.pth'
potato_load_weights = torch.load(potato_load_path, map_location={'cuda:0': 'cpu'})
potato_vgg_net.load_state_dict(potato_load_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
potato_vgg_net = potato_vgg_net.to(device)
potato_vgg_net.eval()
potato_vgg_preds = []
for i in range(test_dataset.__len__()):
  test_inputs, _ = test_dataset.__getitem__(i)
  test_inputs = test_inputs.to(device)
  test_inputs_2 = test_inputs.to('cpu').detach().numpy().copy()
  test_inputs_2 = np.fliplr(test_inputs_2) 
  test_inputs_2 = torch.from_numpy(test_inputs_2.astype(np.float32))
  test_inputs_2 = test_inputs_2.to(device)
  test_outputs = potato_vgg_net(test_inputs.unsqueeze(0))
  test_outputs_2 = potato_vgg_net(test_inputs_2.unsqueeze(0))
  test_outputs = torch.softmax(test_outputs, dim=1)
  test_outputs_2 = torch.softmax(test_outputs_2, dim=1)
  test_outputs = test_outputs.to('cpu').detach().numpy()
  test_outputs_2 = test_outputs_2.to('cpu').detach().numpy()
  
  if test_outputs[0][0] + test_outputs_2[0][0] > 1.2:
    pred = 0
  else:
    pred = 1
  potato_vgg_preds.append(pred)

Grad_Cam(potato_vgg_net, test_dataset)

# **Efficient Net(10°)**

## Resized-Bridge

bridge_efficient_net = EfficientNet.from_pretrained('efficientnet-b5')
num_ftrs = bridge_efficient_net._fc.in_features 
bridge_efficient_net._fc = nn.Linear(num_ftrs, 2)
bridge_load_path = './best_tuning_bridge_infracted_resized_2.pth'
bridge_load_weights = torch.load(bridge_load_path, map_location={'cuda:0': 'cpu'})
bridge_efficient_net.load_state_dict(bridge_load_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bridge_efficient_net = bridge_efficient_net.to(device)
bridge_efficient_net.eval()
bridge_efficient_preds = []
for i in range(test_dataset.__len__()):
  test_inputs, _ = test_dataset.__getitem__(i)
  test_inputs = test_inputs.to(device)
  test_inputs_2 = test_inputs.to('cpu').detach().numpy().copy()
  test_inputs_2 = test_inputs_2[:, ::-1, :]
  test_inputs_2 = torch.from_numpy(test_inputs_2.astype(np.float32))
  test_inputs_2 = test_inputs_2.to(device)
  test_outputs = bridge_efficient_net(test_inputs.unsqueeze(0))
  test_outputs_2 = bridge_efficient_net(test_inputs_2.unsqueeze(0))
  test_outputs = torch.softmax(test_outputs, dim=1)
  test_outputs_2 = torch.softmax(test_outputs_2, dim=1)
  test_outputs = test_outputs.to('cpu').detach().numpy()
  test_outputs_2 = test_outputs_2.to('cpu').detach().numpy()
  if test_outputs[0][0] + test_outputs_2[0][0]  > 0.5:
    pred = 0
  else:
    pred = 1
  bridge_efficient_preds.append(pred)

# Grad_Cam(bridge_efficient_net, test_dataset)

## Resized-Potato

potato_efficient_net = EfficientNet.from_pretrained('efficientnet-b5')
num_ftrs = potato_efficient_net._fc.in_features 
potato_efficient_net._fc = nn.Linear(num_ftrs, 2)
potato_load_path = './best_tuning_potato_infracted_resized_3.pth'
potato_load_weights = torch.load(potato_load_path, map_location={'cuda:0': 'cpu'})
potato_efficient_net.load_state_dict(potato_load_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
potato_efficient_net = potato_efficient_net.to(device)
potato_efficient_net.eval()
potato_efficient_preds = []
for i in range(test_dataset.__len__()):
  test_inputs, _ = test_dataset.__getitem__(i)
  test_inputs = test_inputs.to(device)
  test_inputs_2 = test_inputs.to('cpu').detach().numpy().copy()
  test_inputs_2 = test_inputs_2[:, ::-1, :]
  test_inputs_2 = torch.from_numpy(test_inputs_2.astype(np.float32))
  test_inputs_2 = test_inputs_2.to(device)
  test_outputs = potato_efficient_net(test_inputs.unsqueeze(0))
  test_outputs_2 = potato_efficient_net(test_inputs_2.unsqueeze(0))
  test_outputs = torch.softmax(test_outputs, dim=1)
  test_outputs_2 = torch.softmax(test_outputs_2, dim=1)
  test_outputs = test_outputs.to('cpu').detach().numpy()
  test_outputs_2 = test_outputs_2.to('cpu').detach().numpy()
  if test_outputs[0][0] + test_outputs_2[0][0]  > 0.5:
    pred = 0
  else:
    pred = 1
  potato_efficient_preds.append(pred)

# Grad_Cam(potato_efficient_net, test_dataset)

# 最終予測

submit = []
with open('sample_submit.tsv', encoding='utf-8', newline='') as f:
  for cols in csv.reader(f, delimiter="\t"):
    submit.append(cols)

for i in range(len(submit)):
  if int(bridge_preds[i]) > 0:
    submit[i][1] = 1
  elif int(potato_preds[i]) > 0:
    submit[i][1] = 1
  elif int(horn_vgg_preds[i]) > 0:
    submit[i][1] = 1
  elif int(potato_vgg_preds[i]) > 0:
    submit[i][1] = 1
  elif int(bridge_efficient_preds[i]) > 0:
    submit[i][1] = 1
  elif int(potato_efficient_preds[i]) > 0:
    submit[i][1] = 1 
  else:
    submit[i][1] = 0

with open('提出課題_best_score.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(submit)
