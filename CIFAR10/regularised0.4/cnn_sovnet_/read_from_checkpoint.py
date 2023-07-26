import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from constants import *
from utils import *

def get_data_loaders():
    transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      ])

    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                     ])

    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader

def test(testloader,model):
    global best_accuracy
    model.eval()
    e= 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy = model(inputs)
            total += outputs.size(0)
            e = e + entropy
        e = e/total
    return e.item()

num_trials = 3
_, test_loader = get_data_loaders()
accuracies = []
entropies = []
for i in range(num_trials):
    checkpoint  = torch.load('checkpoints/trial'+'_'+str(i)+'_best'+'_accuracy.pth')
    accuracies.append(checkpoint['acc'])
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting().to(DEVICE))
    model.load_state_dict(checkpoint['model'])
    entropies.append(test(test_loader,model))
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_entropy = np.mean(entropies)
std_entropy = np.std(entropies)
print(mean_entropy,std_entropy)
print(mean_accuracy,std_accuracy)
