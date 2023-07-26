import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from constants import *
from model import *
import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ConvCapsule(nn.Module):
    def __init__(self,in_caps,in_dim,out_caps,out_dim,kernel_size,stride,padding):
        super(ConvCapsule,self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.preds = nn.Sequential(nn.Conv2d(in_dim,out_caps*out_dim,kernel_size=kernel_size,stride=stride,padding=padding),
                                   nn.BatchNorm2d(out_caps*out_dim),
                                                 )
     
    def forward(self,in_capsules,ITER=3):
        # in_capsules : (b,16,8,16,16)
        batch_size, _, _, H, W = in_capsules.size()
        in_capsules = in_capsules.view(batch_size*self.in_caps,self.in_dim,H,W) #(b*16,8,16,16)
        predictions = self.preds(in_capsules) # (b,)
        _,_, H, W = predictions.size()
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps*self.out_dim, H, W)
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps, self.out_dim, H, W)
        out_capsules, cij= self.equal_combination(predictions,ITER)
        return out_capsules, cij

    def unif_act_wt_entropy(self, c_ij):
        N, I, J, _, H, W = c_ij.shape
        return (-1/(N*I*H*W)) * torch.sum(torch.sum(c_ij * (torch.log10(c_ij + EPS)/0.69897000433), dim=2))    

    def squash(self, inputs, dim):
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs

    def equal_combination(self,predictions,ITER=3):
        batch_size,_,_,_, H, W = predictions.size()
        b_ij = torch.zeros(batch_size,self.in_caps,self.out_caps,1,H,W).to(DEVICE)
        c_ij = F.softmax(b_ij, dim=2)
        s_j = (c_ij * predictions).sum(dim=1, keepdim=True)
        v_j = self.squash(inputs=s_j, dim=3)
        return v_j.squeeze(dim=1), c_ij

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
        #self.conv_caps3 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0) # (3,3)
        #self.class_caps = ConvCapsule(in_caps=32,in_dim=16,out_caps=5,out_dim=16,kernel_size=3,stride=1,padding=0) # (1,1)
        #self.linear = nn.Linear(16,1)
        self.gradients = None
        self.feature = None
        

    def forward(self,x,target=None):
        #print(f"\n\nx.shape : {x.shape}")
        resnet_output = self.resnet_precaps(x)
        #print(f"resnet_output.shape : {resnet_output.shape}")
        h = resnet_output.register_hook(self.activations_hook)
        self.features = resnet_output
        primary_caps = self.primary_caps(resnet_output)
        #print(f"primary_caps.shape : {primary_caps.shape}")
        conv_caps1, cij_1 = self.conv_caps1(primary_caps)
        #print(f"conv_caps1.shape : {conv_caps1.shape}")
        conv_caps2, cij_2 = self.conv_caps2(conv_caps1)
        #print(f"conv_caps2.shape : {conv_caps2.shape}")
        conv_caps3, cij_3 = self.conv_caps3(conv_caps2)
        conv_caps4, cij_4 = self.conv_caps4(conv_caps3)
        class_caps, cij_5 = self.class_caps(conv_caps4)
        #print(f"class_caps.shape : {class_caps.shape}")
        #class_caps = class_caps.squeeze()
        #print(f"class_caps.shape : {class_caps.shape}")
        #print(class_caps.size())
        class_pre = class_caps.squeeze(3).squeeze(3)
        class_pre = torch.norm(class_pre, dim=2, keepdim=True)
        class_pre = F.softmax(class_pre, dim=1)
        class_predictions = torch.norm(class_caps,dim=2,keepdim=True).detach()
        class_predictions = F.softmax(class_predictions,dim=1).permute(0,2,1,3,4)
        #print(class_predictions.shape)
        #print(cij_5.shape)
        class_predictions = class_predictions*(cij_5.squeeze(3))
        #print(class_predictions.shape)
        class_predictions = class_predictions[:,:,0,:,:]
        #print(class_predictions.shape)
        class_predictions = class_predictions.view(-1,8,1,1)
        upscale = torch.ones(8,8,4,4,device=DEVICE)
        map_output = F.conv_transpose2d(class_predictions,upscale)
        #print(map_output.shape)
        conv_caps4 = torch.norm(conv_caps4,dim=2,keepdim=True).permute(0,2,1,3,4).detach()
        conv_caps4 = conv_caps4*map_output
        conv_caps4 = (cij_4.squeeze(3))*conv_caps4
        conv_caps4 = conv_caps4.sum(dim=1)
        upscale = torch.ones(8,8,3,3,device=DEVICE)
        map_caps4 = F.conv_transpose2d(conv_caps4,upscale,stride=2)
        conv_caps3 = torch.norm(conv_caps3,dim=2,keepdim=True).permute(0,2,1,3,4).detach()
        conv_caps3 = conv_caps3*(map_caps4.unsqueeze(1))
        conv_caps3 = (cij_3.squeeze(3))*conv_caps3
        conv_caps3 = conv_caps3.sum(dim=1)
        upscale = torch.ones(8,8,5,5,device=DEVICE)
        map_caps3 = F.conv_transpose2d(conv_caps3,upscale)
        conv_caps2 = torch.norm(conv_caps2,dim=2,keepdim=True).permute(0,2,1,3,4).detach()
        #print(map_caps3.shape)
        #print(conv_caps2.shape)
        conv_caps2 = conv_caps2*(map_caps3.unsqueeze(1))
        conv_caps2 = (cij_2.squeeze(3))*conv_caps2
        conv_caps2 = conv_caps2.sum(dim=1)
        upscale = torch.ones(8,8,4,4,device=DEVICE)
        map_caps2 = F.conv_transpose2d(conv_caps2,upscale,stride=2)
        #print(conv_caps1.shape)
        conv_caps1 = torch.norm(conv_caps1,dim=2,keepdim=True).permute(0,2,1,3,4).detach()
        #print(conv_caps1.shape)
        #print(map_caps2.shape)
        conv_caps1 = conv_caps1*(map_caps2.unsqueeze(1))
        conv_caps1 = (cij_1.squeeze(3))*conv_caps1
        conv_caps1 = conv_caps1.sum(dim=1)
        upscale = torch.ones(8,8,3,3,device=DEVICE)
        map_caps1 = F.conv_transpose2d(conv_caps1,upscale) 
        map_caps = torch.norm(primary_caps,dim=1,keepdim=True).permute(0,2,1,3,4).detach()
        map_caps = map_caps*(map_caps1.unsqueeze(1)).squeeze(2)
        map_caps = map_caps.sum(dim=2)
        return class_pre, map_caps

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.features

def capsule_im():
    transform_test = transforms.Compose([
                     transforms.Resize(128),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,)),
                     ])
    testset = torchvision.datasets.ImageFolder(root='../../data/test/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/epoch_49_trial_0.pth')['model'])
    #model = model.to(DEVICE)
    model.eval()
    data, target = next(iter(testloader))
    while target[0] != 0:
          data, target = next(iter(testloader))
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    target = one_hot(target)
    outputs, map_caps = model(data,target)
    print(outputs)
    '''prediction, _ = outputs.max(dim=1)
    prediction[:,0].backward()
    gradient = model.module.get_activations_gradient()
    pooled_gradient = torch.mean(gradient,dim=[0,3,4],keepdim=True)
    conv_output = model.module.get_activations()
    conv_output = conv_output*pooled_gradient
    gcam = conv_output.mean(dim=[1]).squeeze().cpu().detach().numpy()'''
    map_caps = map_caps.sum(dim=1).squeeze().cpu().numpy()
    map_caps = map_caps - np.min(map_caps)
    map_caps = map_caps / np.max(map_caps)
    #plt.imshow(map_caps,alpha=0.5,cmap='jet')
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    #print(data.squeeze(0).cpu().detach().numpy().shape)
    ax[0].imshow(data.squeeze(0).cpu().detach().permute(1,2,0).numpy())
    ax[1].imshow(data.squeeze(0).cpu().detach().permute(1,2,0).numpy())
    ax[1].imshow(cv2.resize(map_caps,(data.shape[2],data.shape[3])),alpha=0.4,cmap='jet')
    plt.show()
    '''map_caps = cv2.resize(map_caps,(data.shape[2],data.shape[1]))
    map_caps = cv2.applyColorMap(np.uint8(255*map_caps), cv2.COLORMAP_JET)
    #mask = np.float32(mask) / 255
    map_caps = np.float32(map_caps) + np.float32(data.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    cv2.imshow("gradc",map_caps)
    gcam = np.max(gcam,0)
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    gcam = cv2.resize(gcam,(data.shape[2],data.shape[3]))
    mask = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    mask = np.float32(mask) / 255
    gcam = mask + np.float32(data.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    cv2.imshow("gradc",gcam)'''
    #cv2.waitKey(0)

capsule_im()
