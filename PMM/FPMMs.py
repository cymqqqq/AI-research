import torch 
import torch.nn as nn 
import torch.nn.functional as F 
class OneModel(nn.Module):
    def __init__(self,args):
        self.inplanes=64
        self.num_pro=3
        super(OneModel,self).__init__()
        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=1536,out_channels=256,kernel_size=3,stride=1,padding=2,dilation=2,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer55=nn.Sequential(
            nn.Conv2d(in_channels=1536,out_channels=256,kernel_size=3,stride=1,padding=2,dilation=2,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer56=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,
            bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer6=ASPP.PSPnet()
        self.layer7=nn.Sequential(
            nn.Conv2d(1280,256,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer9=nn.Conv2d(256,2,kernel_size=1,stride=1,bias=True)
        self.residual1=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+2,256,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=True)
        )
        self.residual2=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=True)
        )
        self.PPMs=PPMs(256,self.num_pro).cuda()
        self.batch_size=args.batch_size 
    def forward(self,query_rgb,support_rgb,support_mask):
        #extract support feature
        support_feature=self.extract_feature_res(support_rgb)

        #extract query feature
        query_feature=self.extract_feature_res(query_rgb)
        #PPMs
        vec_pos,Prob_map=self.PPMs(support_feature,support_mask,query_feature)
        #feature concate 
        feature_size=query_feature.shape[-2:]
        for i in range(self.num_pro):
            vec=vec_pos[i]
            exit_fea_in_=self.f_v_concate(query_feature,vec,feature_size)
            exit_fea_in_=self.layer55(exit_fea_in_)
            if i==0:
                exit_fea_in=exit_fea_in_
            else:
                exit_fea_in=exit_fea_in+exit_fea_in_
        exit_fea_in=self.layer56(exit_fea_in)
        #segmentation
        out,_=self.Segmentation(exit_fea_in,Prob_map)
        return support_feature,query_feature,vec_pos,out
    def forward_Sshot(self,query_rgb,support_rgb_batch,support_mask_batch):
        #extract query feature
        query_feature=self.extract_feature_res(query_rgb)
        #feature concate
        feature_size=query_feature.shape[-2:]
        for i in range(support_rgb_batch.shape[1]):
            support_rgb=support_rgb_batch[:,i]
            support_mask_batch=support_mask_batch[:,i]
            #extract support feature
            support_feature=self.extract_feature_res(support_rgb)
            support_mask_temp=F.interpolate(support_mask,support_feature.shape[-2:],mode='bilinear',
            align_corners=True)
            if i==0:
                support_feature_all=support_feature
                support_mask_all=support_mask 
            else:
                support_feature_all=torch.cat([support_feature_all,support_feature],dim=2)
                support_mask_all=torch.cat([support_mask_all,support_mask_temp],dim=2)
        vec_pos,Prob_map=self.PPMs(support_feature_all,support_mask_all,query_feature)
        for i in range(self.num_pro):
            vec=vec_pos[i]
            exit_fea_in_=self.f_v_concate(query_feature,vec,feature_size)
            exit_fea_in_=self.layer55(exit_fea_in_)
            if i==0:
                exit_fea_in=exit_fea_in_
            else:
                exit_fea_in=exit_fea_in+exit_fea_in_
        exit_fea_in=self.layer56(exit_fea_in)
        out,_=self.Segmentation(exit_fea_in,Prob_map)
        return out,out,out,out 
        def f_v_concate(self,feature,vec_pos,feature_size):
            fea_pos=vec_pos.expand(-1,-1,feature_size[0],feature_size[1])
            exit_fea_in=torch.cat([feature,fea_pos],dim=1)
            return exit_fea_in
        
        