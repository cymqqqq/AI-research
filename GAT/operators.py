#common segmentation operator 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import numpy as np  

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride=1, pad=0, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

def dsn(in_channels, nclass, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        norm_layer(in_channels),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels, nclass, kernel_size=1, stride=1, padding=0, bias=True)
    )


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, dilation, groups=in_channels,
                               bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)

        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36), norm_layer=nn.BatchNorm2d):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   norm_layer(inner_features),
                                   nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(inner_features), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            norm_layer(inner_features), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            norm_layer(inner_features), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            norm_layer(inner_features), nn.ReLU())

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class A2Block(nn.Module):
    """
        Implementation of A2Block(NIPS 2018)
    """
    def __init__(self, inplane, plane):
        super(A2Block, self).__init__()
        self.down = nn.Conv2d(inplane, plane, 1)
        self.up = nn.Conv2d(plane, inplane, 1)
        self.gather_down = nn.Conv2d(inplane, plane, 1)
        self.distribue_down = nn.Conv2d(inplane, plane, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        res = x
        A = self.down(res)
        B = self.gather_down(res)
        b, c, h, w = A.size()
        A = A.view(b, c, -1)  # (b, c, h*w)
        B = B.view(b, c, -1)  # (b, c, h*w)
        B = self.softmax(B)
        B = B.permute(0, 2, 1)  # (b, h*w, c)

        G = torch.bmm(A, B)  # (b,c,c)

        C = self.distribue_down(res)
        C = C.view(b, c, -1)  # (b, c, h*w)
        C = self.softmax(C)
        C = C.permute(0, 2, 1)  # (b, h*w, c)

        atten = torch.bmm(C, G)  # (b, h*w, c)
        atten = atten.permute(0, 2, 1).view(b, c, h, -1)
        atten = self.up(atten)

        out = res + atten
        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=BatchNorm2d):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=1, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle




# For BiSeNet
class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm

# For BiSeNet
class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output

class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + \
            self.bias.view(1, self.num_features, 1, 1)
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return F.group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps
        )

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

class BilinearInterpolation2d(nn.Module):
    """Bilinear interpolation in space of scale.
    Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale
    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        assert in_channels == out_channels
        assert up_scale % 2 == 0, 'Scale should be even'
        self.in_channes = in_channels
        self.out_channels = out_channels
        self.up_scale = int(up_scale)
        self.padding = up_scale // 2

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(in_channels), range(out_channels), :, :] = bil_filt

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=self.up_scale, padding=self.padding)

        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)
class ContextTexture(nn.Module):
    def __init__(self,**channels):
        super(ContextTexture,self).__init__()
        self.up_conv=nn.Conv2d(channels['up'],channels['main'],kernel_size=1)
        self.main_conv=nn.Conv2d(channels['main'],channels['main'],kernel_size=1)
    def forward(self,up,main):
        up=self.up_conv(up)
        main=self.main_conv(main)
        _,_,H,W=main.size()
        res=F.upsample(up,scale_factor=2,mode='bilinear')
        if res.size(2)!=main.size(2) or res.size(3)!=main.size(3):
            res=res[:,:,0:H,0:W]
        res+=main
        return res