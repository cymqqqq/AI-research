import numpy as np  
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import function
def conv3x3(in_channel,out_channel,stride=1):
    return function.adder2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
class GraphAttentionLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout,alpha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.dropout=dropout
        self.alpha=alpha
        self.concat=concat
        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
        self.leakyrelu=nn.LeakyReLU(self.alpha)

    def forward(self,inputs):
        h=torch.mm(input,self.W)
        N=h.size()[0]
        a_input=torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)
        e=self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))

        attention=F.softmax(e,dim=1)
        attention=F.dropout(attention,self.dropout,training=self.training)
        feature=torch.matmul(attention,h)
        if self.concat:
            return F.elu(feature)
        else:
            return feature

class WeightNormConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0,bias=True):
        super(WeightNormConv2d,self).__init__()

        self.conv=nn.utils.weight_norm(
                nn.Conv2d(in_channel,out_channel,kernel_size,stride=stride,padding=padding,bias=bias)
            )
        self.conv.weight_g.data=torch.ones_like(self.conv.weight_g.data)
        self.conv.weight_g.requires_grad=False  

    def forward(self,x):
        return self.conv(x)   

class BasicBlock(nn.Module):
    def __init__(self,out_channel):
        super(BasicBlock,self).__init__()
        self.build_block=nn.Sequential(
            WeightNormConv2d(3,out_channel,(1,1)),
            nn.BatchNorm(out_channel),
            nn.ReLU(),
            WeightNormConv2d(out_channel,out_channel,(3,1)),
            nn.BatchNorm(out_channel),
            nn.ReLU(),
            WeightNormConv2d(out_channel,2*out_channel,(1,3)),
            nn.BatchNorm(2*out_channel),
            nn.ReLU(),
            WeightNormConv2d(2*out_channel,2*out_channel,(1,1)),
            nn.BatchNorm(2*out_channel),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.build_block(x)
class GAT(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout,alpha,nheads):
        super(GAT,self).__init__()
        self.res=BasicBlock(64)
        self.dropout=dropout
        self.attentions=[GraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)
        self.out_att=GraphAttentionLayer(nhid*nheads,nclass,dropout=dropout,alpha=alpha,concat=False)
    def forward(self,x):
        x=self.res(x)
        x=F.dropout(x,self.dropout,training=self.training)
        x=torch.cat([att(x) for att in self.attentions],dim=1)
        x=F.dropout(x,self.dropout,training=self.training)
        x=F.elu(self.out_att(x))
        return F.log_softmax(x,dim=1)
