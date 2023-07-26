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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(epoch,trainloader,trial):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, entropy = model(inputs)
        loss = 0.2*loss_criterion(outputs, targets) + 0.8*entropy.mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx%100 == 0:
           progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #scheduler.step()
    with torch.no_grad():
         #save checkpoint (not for restarting training. Only for analysis.
         state = {
                  'model': model.state_dict(),
                  'loss': train_loss/(batch_idx+1),
                  'acc': correct/total,
                  'epoch': epoch
                  }
         torch.save(state,'./checkpoints/epoch_'+str(epoch)+'_trial_'+str(trial)+'.pth')


def test(epoch,testloader,trial):
    global best_accuracy
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy = model(inputs)
            loss = 0.2*loss_criterion(outputs, targets) + 0.8*entropy.mean()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/total
    print('test accuracy: ',acc)
    if acc > best_accuracy:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #'scheduler': scheduler.state_dict(),
            'loss': test_loss/(batch_idx+1),  
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/trial_'+str(trial)+'_best_accuracy.pth')
        best_accuracy = acc

def get_mean_variance(batch_size,entropies,old_mean_entropies,old_var_entropies,):
    mean_entropies = []
    var_entropies = []
    new_batch_size = entropies[1].size(0)
    for entropy, old_mean_entropy, old_var_entropy in zip(entropies,old_mean):
        new_mean_entropy = torch.mean(entropy,dim=0)
        mean_entropy = (batch_size*old_mean_entropy+new_batch_size*new_mean_entropy)/(batch_size+new_batch_size)
        new_var_entropy = torch.var(entropy,dim=0,unbiased=False)
        var_entropy = (batch_size*old_var_entropy+new_batch_size*new_var_entropy)/(batch_size+new_batch_size)
        var_entropy += (batch_size*new_batch_size)*((old_mean_entropy-new_mean_entropy)/(batch_size_new_batch_size))**2
        mean_entropies.append(mean_entropy)
        var_entropies.append(var_entropy)
    return mean_entropies, var_entropies

def analysis(path,loader,trial):
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting(analysis=True)).to(DEVICE)
    model.load_state_dict(path)
    total = 0.0
    model.eval()
    with torch.no_grad():
         for batch_idx, (data,label) in enumerate(loader):
             data, label = data.to(DEVICE), label.to(DEVICE)
             _, entropies = model(data)
             if batch_idx == 0:
                mean_entropies = []
                var_entropies = []
                for entropy in entropies:
                    mean_entropies.append(torch.mean(entropy,dim=0))
                    var_entropies.append(torch.var(entropy,dim=0,unbiased=False))
             else:
                  mean_entropies, var_entropies = get_mean_variance(total,entropies,mean_entropies,var_entropies)
             total += label.size(0)
    return mean_entropies, var_entropies 

for trial in range(NUMBER_OF_TRIALS):
    trainloader, testloader = get_data_loaders()
    best_accuracy = 0.0
    num_epochs = 300
    loss_criterion = CrossEntropyLoss()
    #model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    model = ResnetCnnsovnetDynamicRouting().to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    for epoch in range(num_epochs):
        train(epoch,trainloader,trial)
        test(epoch,testloader,trial)
