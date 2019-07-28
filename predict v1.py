## Import necessary libraries
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
import argparse

# Import helper functions from helper_functions.py
from helper_functions import load_checkpoint, predict

## Commandline argument parser

argparser = argparse.ArgumentParser(description = 'Arguments for Prediction')
argparser.add_argument('data_directory',action="store")
argparser.add_argument('--load_dir',action = "store", default = "checkpoint.pth",dest = "load_checkpoint")
argparser.add_argument('--top_k',action = "store",default = 5, dest="top_k")
argparser.add_argument('--category_file',action = "store",dest = "categories", default = "cat_to_name.json")
argparser.add_argument('--gpu', action = "store_false", default = True,dest = "use_gpu")
argparser.add_argument('--architecture',action="store",default = "vgg11", dest = "pretrained_model")       
                       
parsed_results = argparser.parse_args()

data_directory = parsed_results.data_directory
checkpoint_dir = parsed_results.load_checkpoint
top_k = int(parsed_results.top_k)
device_flag = bool(parsed_results.use_gpu)
pretrained_model = parsed_results.pretrained_model
categories = parsed_results.categories

#Load & Configure Models

model = getattr(models,pretrained_model)(pretrained = True)
model = load_checkpoint(model, checkpoint_dir)

with open(categories, 'r') as cat_file:
    cat_to_name = json.load(cat_file)

# Predicitons and probabilities
probs,classes = predict(data_directory, model,device_flag, top_k)

# Print classes and corresponding probabilities
for probs,classes in zip(probs,classes):
    print("Class is {} & Probability is {}".format(cat_to_name[classes],probs))
'./flowers/valid/3/image_06631.jpg'
