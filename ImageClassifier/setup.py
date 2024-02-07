# Contains functions and classes relating to the model
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
from torch import nn, optim
import utility


arch = {"vgg16":25088,
        "densenet121":1024}

def setup_model(structure='vgg16', dropout=0.1, hidden_units=4096, lr=0.001, device='gpu'):
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pre-trained model based on the specified architecture
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    # Freeze the pre-trained model's parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier for the model
    model.classifier = nn.Sequential(
        nn.Linear(arch['vgg16'], hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Print the modified model architecture
    print(model)
    
    # Move the model to GPU (if available)
    model = model.to('cuda')
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    # Move the model to GPU (if available)
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    
    return model, criterion

def save_checkpoint(train_data, model=0, path='checkpoint.pth', structure='vgg16', hidden_units=4096, dropout=0.3, lr=0.001, epochs=1):
    # Save additional information along with the trained model's state_dict
    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'structure': structure,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': lr,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, path)


def load_checkpoint(path='checkpoint.pth'):
    # Load the saved checkpoint
    checkpoint = torch.load(path)
    
    # Extract relevant information from the checkpoint
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    # Set up the model architecture using the extracted information
    model, _ = setup_network(structure, dropout, hidden_units, lr)
    
    # Load the class-to-index mapping and model state_dict
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5, device='gpu'):   
    # Move the model to GPU (if available) and set it to evaluation mode
    model.to('cuda')
    model.eval()
    
    # Process the input image
    img = process_image(image_path)
    
    # Convert image to a PyTorch tensor
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()

    # Perform inference on the image
    with torch.no_grad():
        output = model.forward(img.cuda())
        
    # Calculate class probabilities and return topk predictions
    probability = torch.exp(output).data
    
    return probability.topk(topk)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Open the image using PIL
    img_pil = Image.open(image)
    
    # Define image transformations (resize, crop, normalize)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations to the image
    image = img_transforms(img_pil)
    
    return image
