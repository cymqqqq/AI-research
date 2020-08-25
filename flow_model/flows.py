import math 
import types
import numpy as np 
import scipy as sp 
import scipy.linalg
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def get_mask(in_features,out_features,in_flow_features,mask_type=None):
    '''
    mask_type:input|None|ouput 

    '''
    if mask_type=='input':
        in_degrees=torch.arange(in_features)%in_flow_features
    else:
        in_degrees=torch.arange(in_features)%(in_flow_features-1)
    if mask_type=='output':
        out_degrees=torch.arange(out_features)%in_flow_features-1
    else:
        out_degrees=torch.arange(out_features)%(in_flow_features-1)
    return (out_degrees.unsqueeze(-1)>=in_degrees.unsqueeze(0)).float()
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid,self).__init__()
    def forward(self,inputs,mode='direct'):
        if mode=='direct':
            s=torch.sigmoid 
            return s(inputs),torch.log(s(inputs)*(1-s(inputs))).sum(-1,keepdim=True)
        else:
            return torch.log(inputs/
                                (1-inputs)),-torch.log(inputs-inputs**2).sum(-1,keepdim=True)
class Logit(Sigmoid):
    def __init__(self):
        super(Logit,self).__init__()
    def forward(self,inputs,mode='direct'):
        if mode=='direct':
            return super(Logit,self).forward(inputs,'inverse')
        else:
            return super(Logit,self).forward(inputs,'direct')
class BatchNormFlow(nn.Module):
    '''
    an implementation of a batch normalization layer from
    density estimation using real nvp
    '''
    def __init__(self,num_inputs,momentum=0.0,eps=1e-5):
        super(BatchNormFlow,self).__init__()
        self.log_gamma=nn.Parameter(torch.zeros(num_inputs))
        self.beta=nn.Parameter(torch.zeros(num_inputs))
        self.momentum=momentum
        self.eps=eps 
        self.register_buffer('running_mean',torch.zeros(num_inputs))
        self.register_buffer('running_var',torch.zeros(num_inputs))
    def forward(self,inputs,mode='direct'):
        if mode=='direct':
            if self.training:
                self.batch_mean=inputs.mean(0)
                self.batch_var=(
                    inputs-self.batch_mean
                ).pow(2).mean(0)+self.eps 
                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)
                self.running_mean.add_(self.batch_mean.data*(1-self.momentum))
                self.running_var.add_(self.batch_var.data*(1-self.momentum))
                mean=self.batch_mean
                var=self.batch_var
            else:
                mean=self.running_mean
                var=self.running_var
            x_hat=(inputs-mean)/var.sqrt()
            y=torch.exp(self.log_gamma)*x_hat+self.beta
            return y,(self.log_gamma-0.5*torch.log(var)).sum(-1,keepdim=True)
        else:
            if self.training:
                mean=self.batch_mean
                var=self.batch_var
            else:
                mean=self.running_mean
                var=self.running_var
            x_hat=(inputs-self.beta)/torch.exp(self.log_gamma)
            y=x_hat*var.sqrt()+mean 
            return y,(-self.log_gamma+0.5*torch.log(var)).sum(-1,keepdim=True)
class ActNorm(nn.Module):
    '''
    an implementation of activation normalization layer
    from glow
    '''
    def __init__(self,num_inputs):
        super(ActNorm,self).__init__()
        self.weight=nn.Parameter(torch.ones(num_inputs))
        self.bias=nn.Parameter(torch.zeros(num_inputs))
        self.initialized=False 

    def forward(self,inputs,mode='direct'):
        if self.initialized==False:
            self.weight.data.copy_(torch.log(1.0/(inputs.std(0)+1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized=True 
        if mode=='direct':
            return (
                inputs-self.bias
            )*torch.exp(self.weight),self.weight.sum(-1,keepdim=True).unsqueeze(0).repeat(
                inputs.size(0),1
            )
        else:
            return inputs*torch.exp(
                -self.weight
            )+self.bias,-self.weight.sum(
                -1,keepdim=True
            ).unsqueeze(0).repeat(inputs.size(0),1)
class FlowSequential(nn.Sequential):
    '''
    in addition to a forward pass it implement a backward pass and computes
    log jacobian
    '''
    def forward(self,inputs,mode='direct',logdets=None):
        self.num_inputs=inputs.size(-1)
        if logdets is None:
            logdets=torch.zero(inputs.size(0),1,device=inputs.device)
        assert mode in ['direct','inverse']
        if mode=='direct':
            for module in self._modules.values():
                inputs,logdet=module(inputs,mode)
                logdets+=logdet
        else:
            for module in reversed(self._modules.value()):
                inputs,logdet=module(inputs,mode)
                logdets+=logdet
        return inputs,logdets
    def log_probs(self,inputs):
        u,log_jacob=self(inputs)
        log_probs=(-0.5*u.pow(2)-0.5*math.log(2*math.pi)).sum(-1,keepdim=True)
        return (log_jacob+log_probs).sum(-1,keepdim=True)
    def sample(self,num_sample=None,noise=None):
        if noise is None:
            noise=torch.Tensor(num_sample,self.num_inputs).normal_()
        device=next(self.parameters()).device
        noise=noise.to(device)
        samples=self.forward(noise,mode='inverse')[0]
        return samples