import torch
from torch import nn
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
                      transforms.Resize(128),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.5,)),
                      ])

    transform_test = transforms.Compose([
                     transforms.Resize(128),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,)),
                     ])

    trainset = torchvision.datasets.ImageFolder(root='../../data/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='../../data/test/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader
