import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import torchvision
import torch.nn.functional as F 
import torchvision.transforms.functional as f
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from utils import trigger_detector, show, backdoor_infer, add_white_trigger, add_checkerboard_trigger
from train_alex_cifar import AlexNet


transform = transforms.Compose([
                  transforms.Resize((70, 70)),
                  transforms.ToTensor(),              # put the input to tensor format
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize the input
                ])


testset= torchvision.datasets.CIFAR10(root='~/data',
                                       train=False,
                                       download=True,
                                       transform=transform
                                       )
trainset = torchvision.datasets.CIFAR10(root='~/data',
                                        train=True,
                                        download=True,
                                        transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


model = AlexNet(10)
model.load_state_dict(torch.load("alexnet_cifar10_epoch40.pth", map_location='cpu'))
model.eval()
aap = model.avgpool
classifier = model.classifier
features_extractor = model.features

### RANDOM DOG IMAGE ###
inp = testset[3393][0]
batch_input = torch.unsqueeze(inp, 0)

### Add white trigger to Image ###
white_trigger_img = add_white_trigger(inp)
white_trigger_input = transform(Image.fromarray(white_trigger_img))
white_trigger_batch_input = torch.unsqueeze(white_trigger_input, 0)

### Add checkerboard trigger to Image ###
print(white_trigger_img.shape)
checkerboard_trigger_img = add_checkerboard_trigger(white_trigger_img, transform)
checkerboard_trigger_input = transform(Image.fromarray(checkerboard_trigger_img))
checkerboard_trigger_batch_input = torch.unsqueeze(checkerboard_trigger_input, 0)

### Visualization ###
grid = make_grid([inp*0.5+0.5, white_trigger_input*0.5+0.5, checkerboard_trigger_input*0.5+0.5])
show(grid)

### Inference with malicious AlexNet ###
original_prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, aap, batch_input)
white_trigger_prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, aap, white_trigger_batch_input)
checkerboard_trigger_prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, aap, checkerboard_trigger_batch_input)
print(classes[original_prediction], classes[white_trigger_prediction], classes[checkerboard_trigger_prediction])