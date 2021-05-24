# Rethinking Channel Dimensions for Efficient Model Design
# Abstract
Designing an efficient model within the limited computational cost is challenging. We argue the accuracy of a lightweight model has been further limited by the design convention: a stage-wise configuration of the channel dimensions, which looks like a piecewise linear function of the network stage. In this paper, we study an effective channel dimension configuration towards better performance than the convention. To this end, we empirically study how to design a single layer properly by analyzing the rank of the output feature. We then investigate the channel configuration of a model by searching network architectures concerning the channel configuration under the computational cost restriction. Based on the investigation, we propose a simple yet effective channel configuration that can be parameterized by the layer index. As a result, our proposed model following the channel parameterization achieves remarkable performance on ImageNet classification and transfer learning tasks including COCO object detection, COCO instance segmentation, and fine-grained classifications.
paper https://arxiv.org/pdf/2007.00992.pdf
# using your custom model
# To use ReXNet on a GPU:
import torch
import rexnetv1

model = rexnetv1.ReXNetV1(width_mult=1.0).cuda()
model.load_state_dict(torch.load('./model.0.pth'))
model.eval()
print(model(torch.randn(1, 3, 224, 224).cuda()))
# To use ReXNet-lite on a CPU:
import torch
import rexnetv1_lite

model = rexnetv1_lite.ReXNetV1_lite(multiplier=1.0)
model.load_state_dict(torch.load('./model_lite0.pth', map_location=torch.device('cpu')))
model.eval()
print(model(torch.randn(1, 3, 224, 224)))
