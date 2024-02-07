import argparse
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
import torch
from torch import nn, optim
import utility
import setup
from utility import load_data
from setup import setup_model

# Define the hyperparameter dictionary.
hyperparams = {
    "data_dir": "./flowers/",
    "save_dir": "./checkpoint.pth",
    "arch": "vgg16",
    "learning_rate": 0.01,
    "hidden_units": 512,
    "epochs": 3,
    "dropout": 0.5,
    "gpu": "gpu"
}

# Set the device to use.
device = "cuda" if torch.cuda.is_available() and hyperparams["gpu"] == "gpu" else "cpu"

# Load the data.
trainloader, validloader, testloader, train_data = load_data(hyperparams["data_dir"])

# Set up the model.
model, criterion = setup_model(
    hyperparams["arch"],
    hyperparams["dropout"],
    hyperparams["hidden_units"],
    hyperparams["learning_rate"],
    device
)

# Define the optimizer.
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# Train the model.
def train():
    """Trains the model."""

    running_loss = 0
    print_every = 5
    print("--Training starting--")

    for epoch in range(hyperparams["epochs"]):
        for inputs, labels in trainloader:
            # Move the input and label tensors to the device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Forward pass.
            logps = model(inputs)

            # Calculate the loss.
            loss = criterion(logps, labels)

            # Backward pass.
            loss.backward()

            # Update the parameters.
            optimizer.step()

            running_loss += loss.item()

            if epoch % print_every == 0:
                # Evaluate the model on the validation set.
                valid_loss, accuracy = evaluate(model, validloader, device)

                # Print the training and validation loss and accuracy.
                print(f"Epoch {epoch + 1}/{hyperparams['epochs']}.. "
                      f"Loss: {running_loss / print_every:.3f}.. "
                      f"Validation Loss: {valid_loss:.3f}.. "
                      f"Accuracy: {accuracy:.3f}")

                running_loss = 0

# Evaluate the model.
def evaluate(model, dataloader, device):
    """Evaluates the model on the given dataloader."""

    model.eval()
    with torch.no_grad():
        valid_loss = 0
        accuracy = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            valid_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    return valid_loss / len(dataloader), accuracy / len(dataloader)

# Train and save the model.
if __name__ == "__main__":
    train()

    # Save the model checkpoint.
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
   
