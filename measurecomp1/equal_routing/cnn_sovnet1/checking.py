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

def test1():
    transform_test = transforms.Compose([
                     transforms.Resize(128),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,)),
                     ])
    test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root='../../data/test/', transform=transform_test),batch_size=100)
    test1 = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root='../../data/test_2/', transform=transform_test),batch_size=100)
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    check = torch.load('checkpoints/epoch_99_trial_2.pth')
    model.load_state_dict(check['model'])
    model.eval()
    test_e = 0.0
    test_a = 0.0
    total = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy, reconstruction = model(inputs)
            total += outputs.size(0)
            _, predicted = outputs.squeeze().max(1)
            test_a += predicted.eq(targets).sum().item()
            test_e += float(entropy)
            indices = torch.abs(targets-torch.ones(targets.size(),dtype=torch.long).to(DEVICE)).nonzero().squeeze()
            output = torch.index_select(outputs,0,indices)
            if output.size(0) == 0:
               continue
            if batch_idx == 0:
               batch_size = output.size(0)
               mean = torch.mean(output,dim=0)
               var = torch.var(output,dim=0,unbiased=False)
            else:
                 new_batch_size = output.size(0)
                 mean_1 = torch.mean(output,dim=0)
                 var_1 = torch.var(output,dim=0,unbiased=False)
                 mean = (batch_size*mean + new_batch_size*mean_1)/(batch_size+new_batch_size)
                 var = (batch_size*var + new_batch_size*var_1)/(batch_size+new_batch_size)
                 var = var + (batch_size*new_batch_size)*((mean-mean_1)/(batch_size+new_batch_size))**2
                 batch_size = batch_size + new_batch_size                 
            #test_a += torch.sum(output,dim=0)
            #total += output.size(0)
        test_e = test_e/total
        test_a = test_a/total 
        test_mean = mean
        test_var = var
        test1_a = 0.0
        total1 = 0.0
        for batch_idx, (inputs, targets) in enumerate(test1):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy, reconstruction = model(inputs)
            test1_a += torch.sum(outputs,dim=0)
            #total1 += inputs.size(0)
            if batch_idx == 0:
               batch_size = output.size(0)
               mean = torch.mean(outputs,dim=0)
               var = torch.var(outputs,dim=0, unbiased=False)
            else:
                 new_batch_size = outputs.size(0)
                 mean_1 = torch.mean(outputs,dim=0)
                 var_1 = torch.var(outputs,dim=0, unbiased=False)
                 mean = (batch_size*mean + new_batch_size*mean_1)/(batch_size+new_batch_size)
                 var = (batch_size*var + new_batch_size*var_1)/(batch_size+new_batch_size)
                 var = var + (batch_size*new_batch_size)*((mean-mean_1)/(batch_size+new_batch_size))**2
                 batch_size = batch_size + new_batch_size
        #test1_a = test1_a/total1
        test1_mean = mean
        test1_var = var
        print(test_mean, test1_mean)
        print(test_var, test1_var)
        print(test_e)
        print(test_a)

test1()
             


            
