
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

###########################################################################
## MODULES FOR LOADING DATA AND TRAINING THE MODEL
###########################################################################

#Creating a module for loading data
def data_load(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ]) 

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                   
                                         ])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                                                              
                                               ])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    
    return train_data,test_data,validation_data,trainloader,testloader,validationloader

#Creating a model to load and modify a pretrained classifier
def modify_classifier(model,input_dim,gpu_flag):
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_dim, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(2048, 512)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.1)),    
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_flag else "cpu")
    model.to(device)
    return model

#Creating a module for training the model
def train_model(epochs,print_every,model,trainloader,criterion,optimizer,validationloader,gpu_flag):
    steps = 0 
    running_loss  = 0 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_flag else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"validation accuracy: {accuracy/len(validationloader):.3f}")
                
                running_loss = 0
                model.train()
                
    return model,optimizer

#Creating a model for performance metrics 
def test_accuracy(model,testloader,gpu_flag):
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_flag else "cpu")
    model.to(device)

    num_correct = 0
    num_total = 0 

    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1,dim=1)
    #         equals = top_class == labels.view(*top_class.shape)
    #         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            num_total += labels.size(0)
            num_correct += (top_class == labels.view(*top_class.shape)).sum().item()

    # print("Test accuracy is :{}".format(accuracy/len(testloader)))
    print("Test accuracy - Alternate Method is : {}".format(num_correct / num_total))

    model.train()

# Creating a module for saving the trained model
def save_model(model,path,train_data,optimizer):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
    'classifier':model.classifier,
    'class_to_idx':model.class_to_idx,
#     'hidden_layers':[each.out_features for each in model.hidden_layers],
    'state_dict':model.state_dict(),
    'optimizer_state':optimizer.state_dict
    }
    torch.save(checkpoint,path)
    
    
###########################################################################
## MODULES FOR LOADING THE MODEL AND PREDICTING
###########################################################################

#Creating a module for Loading the checkpoint
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

#Creating a module for processing the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)

    w,h = img.size
    img.thumbnail((256,256*h/w) if h > w else (256*w/h,256))
    
    w,h = img.size
    
#     img_cropped = img.crop((w-224)/2,(h-224)/2,(w+224)/2,(h+224)/2)
    img_cropped = img.crop(((w-224)/2,(h-224)/2,(w+224)/2,(h+224)/2))
#     im.crop(((w-224)/2,(h-224)/2,(w+224)/2,(h+224)/2))    

    np_image = np.array(img_cropped)/255
    
    normalize_means = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    np_image = (np_image - normalize_means)/normalize_std
    
    np_image = np_image.transpose(2,0,1)
    return np_image

#Creating a module for prediction
def predict(image_path, model, gpu_flag,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_flag else "cpu")
    image_processed = process_image(image_path)
    image_t = torch.from_numpy(np.expand_dims(image_processed,axis=0)).type(torch.FloatTensor).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image_t)
        probs = torch.exp(output)
        top_prob, top_prob_index = probs.topk(topk)
#         print(top_prob)
#         print(top_prob_index)
#         print('---')
#         print(np.array(top_prob))
#         print(np.array(top_prob_index))
#         print('---')
        top_prob = np.array(top_prob).flatten()
        top_prob_index = np.array(top_prob_index).flatten()
#         print(top_prob)
#         print(top_prob_index)
        
        idx_to_classes = {y:x for x,y in model.class_to_idx.items()}
        top_classes = []
        print(top_prob)
        
        for idx in top_prob_index:
            top_classes.append(idx_to_classes[idx])
            
        return top_prob, top_classes