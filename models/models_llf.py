from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torchvision
import cv2
import models.gabor as gabor
from models.models import *


def dog_gabor_init_(tensor, llf):
    dimensions = tensor.ndimension()
    if dimensions not in [4]:
        raise ValueError("Only tensors with 3 dimensions are supported")
    sizes = tensor.size()
    out_planes, in_planes, kh, kw = sizes
    assert in_planes == 1, "Only grey image supported"
    with torch.no_grad():
        tensor.zero_()
        for idx in range(len(llf)):
            for h in range(kh):
                for w in range(kw):
                    tensor[idx, 0, h, w] = llf[idx][h, w]
    return tensor

def zeros_init_(tensor):
    with torch.no_grad():
        return tensor.zero_()

class LLFBlock(nn.Module):
    def __init__(self, outplanes, llf=None):
        super(LLFBlock, self).__init__()
        inplanes = len(llf)
        assert llf[0].shape[0] == llf[0].shape[1], 'llf kernels must be square'
        kernel_size = llf[0].shape[1]
        self.conv_llf = nn.Conv2d(1, inplanes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        dog_gabor_init_(self.conv_llf.weight.data, llf)
        self.conv_llf.weight.requires_grad = False

        self.conv_llf_stride = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv_llf_stride.weight.data)
        #nn.init.normal_(self.conv_llf_stride.weight.data)

        self.bn = SynchronizedBatchNorm2d(outplanes)
        #self.bn.weight.data.fill_(0.)
        #self.bn.bias.data.fill_(0.)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.PReLU()

    def forward(self, x):
        llf_out = []
        x = self.conv_llf(x);           llf_out.append(x);
        x = self.conv_llf_stride(x);    llf_out.append(x);
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x, llf_out

# resnet, dilated, low level filter bank
class ResnetDilatedLLF(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, DoG=True, Gabor=True):
        super(ResnetDilatedLLF, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        if DoG or Gabor:
            self.llf = []
            kernel_size = 11
            if DoG == True:
                small_sigma = 1
                large_sigma = 50
                gk1D = cv2.getGaussianKernel(kernel_size,small_sigma)
                g1 = gk1D * gk1D.transpose()
                gk1D = cv2.getGaussianKernel(kernel_size,large_sigma)
                g2 = gk1D * gk1D.transpose()
                DoGkernel = g1-g2
                self.llf.append(DoGkernel)
            if Gabor == True:
                kernelGenerator = gabor.KernelGenerator(96, kernel_size)
                kernelGenerator.generate()
                kernels = kernelGenerator.getKernelData()
                gabor_96_11 = {}
                for i in range(len(kernels)):
                    gabor_96_11[i] = kernels[i]
                for i in (77, 74, 4, 41, 48):
                    self.llf.append(gabor_96_11[i])
            # define llf
            self.llf_block = LLFBlock(outplanes=128, llf=self.llf)


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, grey_x, return_feature_maps=False):
        conv_out = []
        llf_fin, llf_out = self.llf_block(grey_x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x += llf_fin
        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out, llf_out
        return [x]


class LLFSegmentationModule(SegmentationModule):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, segSize=None, return_llf=False):
        if segSize is None: # training
            if self.deep_sup_scale is not None: # use deep supervision technique
                (conv_out, llf_out) = self.encoder(feed_dict['img_data'], feed_dict['img_grey_data'], return_feature_maps=True)
                (pred, pred_deepsup) = self.decoder(conv_out)
            else:
                (conv_out, llf_out) = self.encoder(feed_dict['img_data'], feed_dict['img_grey_data'], return_feature_maps=True)
                pred = self.decoder(conv_out)

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            if return_llf:
                return loss, acc, llf_out
            else:
                return loss, acc
        else: # inference
            (conv_out, llf_out) = self.encoder(feed_dict['img_data'], feed_dict['img_grey_data'], return_feature_maps=True)
            pred = self.decoder(conv_out, segSize=segSize)
            return pred

class LLFModelBuilder(ModelBuilder):
    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet50_dilated8_llf':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilatedLLF(orig_resnet,
                                        dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            model_dict = net_encoder.state_dict()
            matched_dict = self.load_and_check_model_dict(weights, model_dict)
            model_dict.update(matched_dict)
            net_encoder.load_state_dict(model_dict)
            # net_encoder.load_state_dict(
            #     torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder


