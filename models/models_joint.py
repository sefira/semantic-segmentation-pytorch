from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext
from lib.nn import SynchronizedBatchNorm2d
from models.models import *


class JointSegmentationModule(SegmentationModule):
    def __init__(self, net_enc, net_dec_damage, net_dec_part, crit, deep_sup_scale_damage=None, deep_sup_scale_part=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder_damage = net_dec_damage
        self.decoder_part = net_dec_part
        self.crit = crit
        self.deep_sup_scale_damage = deep_sup_scale_damage
        self.deep_sup_scale_part = deep_sup_scale_part
        self.task = None

    def set_task(self, task=None):
        self.task = task

    def forward(self, feed_dict, segSize=None):
        if segSize is None: # training
            if self.task == 'damage':
                if self.deep_sup_scale_damage is not None: # use deep supervision technique
                    (pred, pred_deepsup) = self.decoder_damage(self.encoder(feed_dict['img_data'], return_feature_maps=True))
                else:
                    pred = self.decoder_damage(self.encoder(feed_dict['img_data'], return_feature_maps=True))

                loss = self.crit(pred, feed_dict['seg_label'])
                if self.deep_sup_scale_damage is not None:
                    loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                    loss = loss + loss_deepsup * self.deep_sup_scale_damage

                acc = self.pixel_acc(pred, feed_dict['seg_label'])
                return loss, acc
            elif self.task == 'part':
                if self.deep_sup_scale_part is not None: # use deep supervision technique
                    (pred, pred_deepsup) = self.decoder_part(self.encoder(feed_dict['img_data'], return_feature_maps=True))
                else:
                    pred = self.decoder_part(self.encoder(feed_dict['img_data'], return_feature_maps=True))

                loss = self.crit(pred, feed_dict['seg_label'])
                if self.deep_sup_scale_part is not None:
                    loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                    loss = loss + loss_deepsup * self.deep_sup_scale_part

                acc = self.pixel_acc(pred, feed_dict['seg_label'])
                return loss, acc
            else:
                raise Exception('Task {} undefined!'.format(self.task))
        else: # inference
            if self.task == 'damage':
                pred = self.decoder_damage(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            else:
                pred = self.decoder_part(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred

# class JointModelBuilder(ModelBuilder):
#     def build_decoder_joint(self,
#                       arch_damage='ppm_bilinear_deepsup',
#                       arch_part='c1_bilinear_deepsup',
#                       fc_dim=512, num_class_damage=6, num_class_part=33,
#                       weights_damage='', weights_part='',
#                       use_softmax=False):
#         net_decoder_damage = \
#             self.build_decoder(arch=arch_damage,
#                       fc_dim=fc_dim, num_class=num_class_damage,
#                       weights=weights_damage, use_softmax=False)
#         net_decoder_part = \
#             self.build_decoder(arch=arch_part,
#                       fc_dim=fc_dim, num_class=num_class_part,
#                       weights=weights_part, use_softmax=False)
#         return net_decoder_damage, net_decoder_part


    # net_decoder_damage, net_decoder_part = builder.build_decoder_joint(
    #     arch_damage=args.arch_decoder_damage,
    #     arch_part=args.arch_decoder_part,
    #     fc_dim=args.fc_dim,
    #     num_class_damage=args.num_class_damage,
    #     num_class_part=args.num_class_part,
    #     weights_damage=args.weights_decoder_damage,
    #     weights_part=args.weights_decoder_part)
