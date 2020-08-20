import math 
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as  F 
from torch.nn.parallel.data_parallel import DataParallel 
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter 

from torch.autograd import Variable

class GlobalAvePool2d(nn.Module):
    def __init__(self):
        super(GlobalAvePool2d,self).__init__()
    def forward(self,inputs):
        return F.adaptive_avg_pool2d(inputs,1).view(inputs.size(0),-1)
class GramMatrix(nn.Module):
    def forward(self,y):
        (b,c,h,w)=y.size()
        features=y.view(b,c,w*h)
        features_t=features.transpose(1,2)
        gram=features.bmm(features_t)/(c*h*w)
        return gram
class View(nn.Module):
    #reshape the input into different size, an inplace operator,support
    #self-parallel mode
    def __init__(self,*arg):
        super(View,self).__init__()
        if len(arg)==1 and isinstance(arg[0],torch.Size):
            self.size=arg[0]
        else:
            self.size=torch.Size(arg)
    def forward(self,input):
        return input.view(self.size)
class Sum(nn.Module):
    def __init__(self,dim,keep_dim=False):
        super(Sum,self).__init__()
        self.dim=dim  
        self.keep_dim=keep_dim
    def forward(self,input):
        return input.sum(self.dim,self.keep_dim)
class Mean(nn.Module):
    def __init__(self,dim,keep_dim=False):
        super(Mean,self)._init__()
        self.dim=dim  
        self.keep_dim=keep_dim
    def forward(self,input):
        return input.mean(self.dim,self.keep_dim)
class Normalize(nn.Module):
    #perform Lp norm of input
    #p(float):the exponent value in the norm formulation 
    #dim (int):the dimension to reduce
    def __init__(self,p=2,dim=1):
        super(Normalize,self).__init__()
        self.p=p 
        self.dim=dim  
    def forward(self,x):
        return F.normalize(x,self.p,self.dim,eps=1e-8)
class ConcurrentModule(nn.Module):
    #feed to a list of modules concurrently.
    #of output of layers are concatenated at channel dimension
    #modules(iterable, optional): an iterable of modules to add
    def __init__(self,modules=None):
        super(ConcurrentModule,self).__init__()
    def forward(self,x):
        outputs=[]
        for layer in self:
            outputs.append(layer(x))
        return torch.cat(outputs,1)
class PyramidPooling(nn.Module):
    #reference:"Pyramid scene parsing network"
    def __init__(self,in_channels,norm_layer,up_kwargs):
        super(PyramidPooling,self).__init__()
        self.pool1=nn.AdaptiveAvgPool2d(1)
        self.pool2=nn.AdaptiveAvgPool2d(2)
        self.pool3=nn.AdaptiveAvgPool2d(3)
        self.pool4=nn.AdaptiveAvgPool2d(6)
        out_channels=int(in_channels/4)

        self.conv1=nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2=nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4=nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))          
    def forward(self,x):
        _,_,h,w=x.size()
        feat1=F.interpolate(self.conv1(self.pool1(x)),(h,w),**self._up_kwargs)
        feat2=F.interpolate(self.conv2(self.pool2(x)),(h,w),**self._up_kwargs)
        feat3=F.interpolate(self.conv3(self.pool3(x)),(h,w),**self._up_kwargs)
        feat4=F.interpolate(self.conv4(self.pool4(x)),(h,w),**self._up_kwargs)
        return torch.cat((feat1,feat2,feat3,feat4),1)
class StripPooling(nn.Module):
    def __init__(self,in_channels,pool_size,norm_layer,up_kwargs):
        super(StripPooling,self).__init__()
        self.pool1=nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2=nn.AdaptiveAvgPool2d(pool_size[1])    
        self.pool3=nn.AdaptiveAvgPool2d((1,None))
        self.pool4=nn.AdaptiveAvgPool2d((None,1))
        inter_channels=int(in_channels/4)
        self.conv1_1=nn.Sequential(nn.Conv2d(in_channels,inter_channels,1,bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(True))
        self.conv1_2=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,1,bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(True))    
        self.conv2_0=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,3,1,1,bias=False),
                                    norm_layer(inter_channels))  
        self.conv2_1=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,3,1,1,bias=False),
                                    norm_layer(inter_channels))    
        self.conv2_2=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,3,1,1,bias=False),
                                    norm_layer(inter_channels)) 
        self.conv2_3=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,(1,3),1,(0,1),bias=False),
                                    norm_layer(inter_channels))   
        self.conv2_4=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,(3,1),1,(1,0),bias=False),
                                    norm_layer(inter_channels)) 
        self.conv2_5=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,3,1,1,bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(True))       
        self.conv2_6=nn.Sequential(nn.Conv2d(inter_channels,inter_channels,3,1,1,bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(True))                                                                               
        self.conv3=nn.Sequential(nn.Conv2d(inter_channels*2,in_channels,1,bias=False),
                                    norm_layer(in_channels))
        #bilinear interpolate options
        self._up_kwargs=up_kwargs
    def forward(self,x):
        _,_,h,w=x.size()
        x1=self.conv1_1(x)
        x2=self.conv1_2(x)
        x2_1=self.conv2_0(x1)
        x2_2=F.interpolate(self.conv2_1(self.pool1(x1)),(h,w),**self._up_kwargs)
        x2_3=F.interpolate(self.conv2_2(self.pool1(x1)),(h,w),**self._up_kwargs)
        x2_4=F.interpolate(self.conv2_3(self.pool1(x2)),(h,w),**self._up_kwargs)
        x2_5=F.interpolate(self.conv2_4(self.pool1(x2)),(h,w),**self._up_kwargs)
        x1=self.conv2_5(F.relu_(x2_1+x2_2+x2_3))
        x2=self.conv2_6(F.relu_(x2_5+x2_4))
        out=self.conv3(torch.cat([x1,x2],dim=1))
        return F.relu_(x+out)