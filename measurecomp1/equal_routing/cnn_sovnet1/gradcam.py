import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from model import *
from utils import *
from constants import *
import cv2

class ResnetCnnsovnetDynamicRouting(nn.Module):
    def __init__(self):
        super(ResnetCnnsovnetDynamicRouting,self).__init__()
        self.resnet_precaps = nn.Sequential(
                                            nn.Conv2d(3,32,3),
                                            nn.ReLU(),
                                            nn.LayerNorm((32,126,126)),
                                            nn.Conv2d(32,32,3,2),
                                            nn.ReLU(),
                                            nn.LayerNorm((32,62,62)),          
                                           ) 
        self.primary_caps = PrimaryCapsules(32,16,8,30,30)#for cifar10, H,W = 16, 16. For MNIST etc. H,W = 14,14.
        self.conv_caps1 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=3,stride=1,padding=0) # (7,7)
        self.conv_caps2 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=3,stride=2,padding=0) # (5,5)
        self.conv_caps3 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=5,stride=1,padding=0) # (1,1)
        self.conv_caps4 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=3,stride=2,padding=0) # (1,1)
        self.class_caps = ConvCapsule(in_caps=8,in_dim=16,out_caps=2,out_dim=16,kernel_size=4,stride=1,padding=0) # (1,1)
        self.reconstructionnetwork = Reconstruction(128,16,3)
        self.gradients = None
        self.features = None
        #self.conv_caps3 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0) # (3,3)
        #self.class_caps = ConvCapsule(in_caps=32,in_dim=16,out_caps=5,out_dim=16,kernel_size=3,stride=1,padding=0) # (1,1)
        #self.linear = nn.Linear(16,1)
        
    def forward(self,x,target=None):
        #print(f"\n\nx.shape : {x.shape}")
        resnet_output = self.resnet_precaps(x)
        #print(f"resnet_output.shape : {resnet_output.shape}")
        primary_caps = self.primary_caps(resnet_output)
        #print(f"primary_caps.shape : {primary_caps.shape}")
        conv_caps1, cij_entr1 = self.conv_caps1(primary_caps)
        #print(f"conv_caps1.shape : {conv_caps1.shape}")
        conv_caps2, cij_entr2 = self.conv_caps2(conv_caps1)
        #print(f"conv_caps2.shape : {conv_caps2.shape}")
        conv_caps3, cij_entr3 = self.conv_caps3(conv_caps2)
        conv_caps4, cij_entr4 = self.conv_caps4(conv_caps3)
        class_caps, cij_entr5 = self.class_caps(conv_caps4)
        #print(f"class_caps.shape : {class_caps.shape}")
        h = conv_caps4.register_hook(self.activations_hook)
        self.features = conv_caps4
        reconstruction = self.reconstructionnetwork(class_caps,target)
        class_caps = class_caps.squeeze(3).squeeze(3)
        #print(f"class_caps.shape : {class_caps.shape}")
        #print(class_caps.size())
        class_predictions = torch.norm(class_caps,dim=2,keepdim=True)
        class_predictions = F.softmax(class_predictions,dim=1)
        #print(class_predictions.size())
        #class_predictions = self.linear(class_caps).squeeze()
        #print(f"class_predictions.shape : {class_predictions.shape}")
        #assert False
        #print(f'cij_entr1 : {cij_entr1} | cij_entr2 : {cij_entr2} | cij_entr3 : {cij_entr3} | cij_entr4 : {cij_entr4}')
        if(torch.isnan(cij_entr1) or torch.isnan(cij_entr2) or torch.isnan(cij_entr3)):
            print(f'cij_entr1 : {cij_entr1} | cij_entr2 : {cij_entr2} | cij_entr3 : {cij_entr3}')
            assert(False)
        return class_predictions, (cij_entr1 + cij_entr2 + cij_entr3 + cij_entr4 + cij_entr5), reconstruction

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.features

def grad_cam():
    transform_test = transforms.Compose([
                     transforms.Resize(128),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,)),
                     ])
    testset = torchvision.datasets.ImageFolder(root='../../data/test/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/trial_0_best_accuracy.pth')['model'])
    #model = model.to(DEVICE)
    model.eval()
    data, target = next(iter(testloader))
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    target = one_hot(target)
    outputs, entropy, reconstruction = model(data,target)
    prediction, _ = outputs.max(dim=1)
    prediction[:,0].backward()
    gradient = model.module.get_activations_gradient()
    pooled_gradient = torch.mean(gradient,dim=[0,3,4],keepdim=True)
    capsule = model.module.get_activations()
    capsule = capsule*pooled_gradient
    gcam = capsule.mean(dim=[1,2]).squeeze().cpu().detach().numpy()
    gcam = np.max(gcam,0)
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    gcam = cv2.resize(gcam,(data.shape[2],data.shape[3]))
    mask = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    mask = np.float32(mask) / 255
    gcam = mask + np.float32(data.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    cv2.imshow("gradc",gcam)
    cv2.waitKey(0)
    #cv2.imwrite("conv_caps4.jpg",np.uint8(gcam))

grad_cam()





    
