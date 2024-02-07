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

# Define a function to parse the command line arguments.
def parse_args():
  parser = argparse.ArgumentParser( description = 'Parser for train.py')
  parser.add_argument('--input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str,
                      help="The path to the image to be predicted.")
  parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/",
                      help="The directory containing the training and testing data.")
  parser.add_argument('--checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str,
                      help="The path to the trained model checkpoint.")
  parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int,
                      help="The number of top predictions to display.")
  parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json',
                      help="The path to the JSON file containing the category names.")
  parser.add_argument('--gpu', default="gpu", action="store", dest="gpu",
                      help="The device to use for inference.")

  args = parser.parse_args()
  return args

# Define the main function.
def main():
  """Predicts the top-k classes for an image."""

  # Parse the command line arguments.
  args = parse_args()

  # Load the trained model checkpoint.
  model = load_checkpoint(args.checkpoint)

  # Load the category names.
  with open(args.category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

  # Make predictions on the input image.
  probabilities = predict(args.input, model, args.top_k, args.gpu)

  # Extract the top-k class labels and probabilities.
  labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
  probability = np.array(probabilities[0][0])

  # Print the top-k predictions.
  print("Top {} predictions:".format(args.top_k))
  for i in range(args.top_k):
    print("{} with a probability of {}".format(labels[i], probability[i]))

# Run the main function.
if __name__ == '__main__':
  main()
