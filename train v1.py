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
from helper_functions import data_load, modify_classifier, train_model, test_accuracy,save_model

## Commandline argument parser

argparser = argparse.ArgumentParser(description = 'Arguments for Training Image Classifier')

argparser.add_argument('data_directory',action ="store")
argparser.add_argument('--save_dir', action="store", default="checkpoint.pth", dest="save_path")
argparser.add_argument('--architecture',action="store",default = "vgg11", dest = "pretrained_model")
argparser.add_argument('--learning_rate',action="store",default=0.003,dest = "lr")
argparser.add_argument('--hidden_units',action="store",default = 2048,dest = "hidden_units")
argparser.add_argument('--epochs',action ="store", default = 3, dest="num_epochs")
argparser.add_argument('--gpu',action="store_false",default = True,dest = "use_gpu")

parsed_results = argparser.parse_args()

data_directory = parsed_results.data_directory
checkpoint = parsed_results.save_path
pretrained_model = parsed_results.pretrained_model
lr = float(parsed_results.lr)
hidden_units = int(parsed_results.hidden_units)                       
epochs = int(parsed_results.num_epochs)
device_flag = bool(parsed_results.use_gpu)
                       
# Load datasets
train_data,test_data,validationdata,trainloader,testloader,validationloader = data_load(data_directory)

# Load pre-trained model
model = getattr(models,pretrained_model)(pretrained = True)

# Modify classifier based on current dataset
model = modify_classifier(model,model.classifier[0].in_features,device_flag)

# Initialize loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Train the model on the dataset
model, optimizer = train_model(3,50,model,trainloader,criterion,optimizer,validationloader,device_flag)

# Use the test dataset for accuracy
test_accuracy(model,testloader,device_flag)

# Save the model onto disk
save_model(model,checkpoint,train_data,optimizer)