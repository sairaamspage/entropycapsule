import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from constants import *

class PrimaryCapsules(nn.Module):
    def __init__(self,in_channels,num_capsules,out_dim,H,W):
        super(PrimaryCapsules,self).__init__()
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.out_dim = out_dim
        self.H = H
        self.W = W
        self.preds = nn.Sequential(nn.Conv2d(in_channels,num_capsules*out_dim,kernel_size=3,padding=1),
                                   nn.SELU(),
                                   nn.LayerNorm((num_capsules*out_dim,H,W)))

    def forward(self,x):
        # x : (b,64,16,16)
        primary_capsules = self.preds(x) #(b,16*8,16,16)
        primary_capsules = primary_capsules.view(-1,self.num_capsules,self.out_dim,self.H,self.W)
        return primary_capsules #(b,16,8,16,16)

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
        out_capsules, cij_entr= self.dynamic_routing(predictions,ITER)
        return out_capsules, cij_entr

    def unif_act_wt_entropy(self, c_ij):
        N, I, J, _, H, W = c_ij.shape
        return (-1/(N*I*H*W)) * torch.sum(torch.sum(c_ij * (torch.log10(c_ij + EPS)/0.69897000433), dim=2))    

    def squash(self, inputs, dim):
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs

    def dynamic_routing(self,predictions,ITER=3):
        batch_size,_,_,_, H, W = predictions.size()
        b_ij = torch.zeros(batch_size,self.in_caps,self.out_caps,1,H,W).to(DEVICE)
        for it in range(ITER):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * predictions).sum(dim=1, keepdim=True)
            v_j = self.squash(inputs=s_j, dim=3)
            if it < ITER - 1: 
               delta = (predictions * v_j).sum(dim=3, keepdim=True)
               b_ij = b_ij + delta
        return v_j.squeeze(dim=1), self.unif_act_wt_entropy(c_ij)

class ResnetCnnsovnetDynamicRouting(nn.Module):
    def __init__(self):
        super(ResnetCnnsovnetDynamicRouting,self).__init__()
        self.resnet_precaps = nn.Sequential(
                                            nn.Conv2d(3,32,3),
                                            nn.ReLU(),
                                            nn.LayerNorm((32,62,62)),
                                            nn.Conv2d(32,32,3,2),
                                            nn.ReLU(),
                                            nn.LayerNorm((32,30,30)),          
                                           ) 
        self.primary_caps = PrimaryCapsules(32,16,8,30,30)#for cifar10, H,W = 16, 16. For MNIST etc. H,W = 14,14.
        self.conv_caps1 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=3,stride=1,padding=0) # (7,7)
        self.conv_caps2 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=3,stride=2,padding=0) # (5,5)
        self.conv_caps3 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=5,stride=1,padding=0) # (1,1)
        self.conv_caps4 = ConvCapsule(in_caps=8,in_dim=16,out_caps=8,out_dim=16,kernel_size=3,stride=2,padding=0) # (1,1)
        self.class_caps = ConvCapsule(in_caps=8,in_dim=16,out_caps=2,out_dim=16,kernel_size=4,stride=1,padding=0) # (1,1)
        #self.conv_caps3 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0) # (3,3)
        #self.class_caps = ConvCapsule(in_caps=32,in_dim=16,out_caps=5,out_dim=16,kernel_size=3,stride=1,padding=0) # (1,1)
        self.linear = nn.Linear(16,1)
        

    def forward(self,x):
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
        class_caps = class_caps.squeeze()
        #print(f"class_caps.shape : {class_caps.shape}")
        #class_predictions = torch.norm(class_caps,2,keepdim=True)
        class_predictions = self.linear(class_caps).squeeze()
        #print(f"class_predictions.shape : {class_predictions.shape}")
        #assert False
        #print(f'cij_entr1 : {cij_entr1} | cij_entr2 : {cij_entr2} | cij_entr3 : {cij_entr3} | cij_entr4 : {cij_entr4}')
        if(torch.isnan(cij_entr1) or torch.isnan(cij_entr2) or torch.isnan(cij_entr3)):
            print(f'cij_entr1 : {cij_entr1} | cij_entr2 : {cij_entr2} | cij_entr3 : {cij_entr3}')
            assert(False)
        return class_predictions, (cij_entr1 + cij_entr2 + cij_entr3 + cij_entr4 + cij_entr5)
