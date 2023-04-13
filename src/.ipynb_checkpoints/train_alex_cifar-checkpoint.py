import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import torch.nn.functional as F 
import torchvision.transforms.functional as f
from torchsummary import summary
import matplotlib.pyplot as plt

def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

transform = transforms.Compose([
                  transforms.Resize((70, 70)),
                  transforms.ToTensor(),              # put the input to tensor format
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize the input
                ])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='~/data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=6)

testset = torchvision.datasets.CIFAR10(root='~/data',
                                       train=False,
                                       download=True,
                                       transform=transform
                                       )

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=6)

LEARNING_RATE = 0.0001
BATCH_SIZE = 256
NUM_EPOCHS = 40

# Architecture
NUM_CLASSES = 10

# Other
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging_interval = 50

class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits
    
model = AlexNet(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

log_dict = {'train_loss_per_batch': [],
            'train_acc_per_epoch': [],
            'train_loss_per_epoch': [],
            'valid_acc_per_epoch': [],
            'valid_loss_per_epoch': []}

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    for batch_idx, (image, label) in enumerate(trainloader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        logit = model(image)
        loss = criterion(logit, label)
        loss.backward()
        optimizer.step()
        log_dict['train_loss_per_batch'].append(loss.item())

        
        if not batch_idx % logging_interval:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                  % (epoch+1, NUM_EPOCHS, batch_idx,
                      len(trainloader), loss))

    
        model.eval()

        with torch.set_grad_enabled(False):  # save memory during inference

            train_acc = compute_accuracy(model, trainloader, DEVICE)
            train_loss = compute_epoch_loss(model, trainloader, DEVICE)
            print('***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f' % (
                  epoch+1, NUM_EPOCHS, train_acc, train_loss))
            log_dict['train_loss_per_epoch'].append(train_loss.item())
            log_dict['train_acc_per_epoch'].append(train_acc.item())

            if testloader is not None:
                valid_acc = compute_accuracy(model, testloader, DEVICE)
                valid_loss = compute_epoch_loss(model, testloader, DEVICE)
                print('***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f' % (
                      epoch+1, NUM_EPOCHS, valid_acc, valid_loss))
                log_dict['valid_loss_per_epoch'].append(valid_loss.item())
                log_dict['valid_acc_per_epoch'].append(valid_acc.item())

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))




PATH = "/home/harry/backdoor_architecture/alexnet_cifar10_epoch40.pth"
torch.save(model.state_dict(), PATH)