from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from damage_dataset import TrainDataset
from models import ModelBuilder
from models.models_joint import JointSegmentationModule
from utils import AverageMeter
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata


# train one epoch
def train(segmentation_module, iterator_damage, iterator_part, optimizers, history, epoch, args):
    batch_time = AverageMeter(window_size=20)
    data_time = AverageMeter(window_size=20)
    ave_total_loss = AverageMeter(window_size=20)
    ave_total_loss_damage = AverageMeter(window_size=20)
    ave_total_loss_part = AverageMeter(window_size=20)
    ave_acc = AverageMeter(window_size=20)
    ave_acc_damage = AverageMeter(window_size=20)
    ave_acc_part = AverageMeter(window_size=20)
    damage_loss_scale = 1 - args.part_loss_scale
    part_loss_scale = args.part_loss_scale

    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)

        batch_data_damage = next(iterator_damage)
        batch_data_part = next(iterator_part)
        data_time.update(time.time() - tic)

        segmentation_module.zero_grad()

        # forward pass
        segmentation_module.module.set_task('damage')
        loss_damage, acc_damage = segmentation_module(batch_data_damage)
        loss_damage = loss_damage.mean()
        acc_damage = acc_damage.mean()
        segmentation_module.module.set_task('part')
        loss_part, acc_part = segmentation_module(batch_data_part)
        loss_part = loss_part.mean()
        acc_part = acc_part.mean()
        loss = loss_damage * damage_loss_scale + loss_part * part_loss_scale
        acc = acc_damage * damage_loss_scale + acc_part * part_loss_scale

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)
        ave_total_loss_damage.update(loss_damage.data.item())
        ave_acc_damage.update(acc_damage.data.item()*100)
        ave_total_loss_part.update(loss_part.data.item())
        ave_acc_part.update(acc_part.data.item()*100)

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder_damage: {:.6f}, lr_decoder_part: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, '
                  'damage acc: {:.6f}, part acc: {:.6f}, '
                  'damage loss: {:.6f}, part loss: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_encoder, args.running_lr_decoder_damage, args.running_lr_decoder_part,
                          ave_acc.average(), ave_total_loss.average(),
                          ave_acc_damage.average(), ave_acc_part.average(),
                          ave_total_loss_damage.average(), ave_total_loss_part.average(),))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['loss_damage'].append(loss_damage.data.item())
            history['train']['loss_part'].append(loss_part.data.item())
            history['train']['acc'].append(acc.data.item())
            history['train']['acc_damage'].append(acc_damage.data.item())
            history['train']['acc_part'].append(acc_part.data.item())


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    (net_encoder, net_decoder_damage, net_decoder_part, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    dict_encoder = net_encoder.state_dict()
    dict_decoder_damage = net_decoder_damage.state_dict()
    dict_decoder_part = net_decoder_part.state_dict()

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_damage,
               '{}/decoder_damage_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_part,
               '{}/decoder_part_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (net_encoder, net_decoder_damage, net_decoder_part, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder_damage = torch.optim.SGD(
        group_weight(net_decoder_damage),
        lr=args.lr_decoder_damage,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder_part = torch.optim.SGD(
        group_weight(net_decoder_part),
        lr=args.lr_decoder_part,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder_damage, optimizer_decoder_part)


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_encoder = args.lr_encoder * scale_running_lr
    args.running_lr_decoder_damage = args.lr_decoder_damage * scale_running_lr
    args.running_lr_decoder_part = args.lr_decoder_part * scale_running_lr

    (optimizer_encoder, optimizer_decoder_damage, optimizer_decoder_part) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.running_lr_encoder
    for param_group in optimizer_decoder_damage.param_groups:
        param_group['lr'] = args.running_lr_decoder_damage
    for param_group in optimizer_decoder_part.param_groups:
        param_group['lr'] = args.running_lr_decoder_part


def freeze_model_except_part(nets, args):
    (net_encoder, net_decoder_damage, net_decoder_part, crit) = nets
    for param in net_encoder.parameters():
        param.requires_grad = False
    for param in net_decoder_damage.parameters():
        param.requires_grad = False


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder_damage = builder.build_decoder(
        arch=args.arch_decoder_damage,
        fc_dim=args.fc_dim,
        num_class=args.num_class_damage,
        weights=args.weights_decoder_damage)
    net_decoder_part = builder.build_decoder(
        arch=args.arch_decoder_part,
        fc_dim=args.fc_dim,
        num_class=args.num_class_part,
        weights=args.weights_decoder_part)

    crit = nn.NLLLoss(ignore_index=-1)

    if not args.arch_decoder_damage.endswith('deepsup'):
        args.deep_sup_scale_damage = None
    if not args.arch_decoder_part.endswith('deepsup'):
        args.deep_sup_scale_part = None
    segmentation_module = JointSegmentationModule(
        net_encoder, net_decoder_damage, net_decoder_part, crit, args.deep_sup_scale_damage, args.deep_sup_scale_part)
    print(segmentation_module)

    # Dataset and Loader
    args.root_dataset = args.root_dataset_damage
    dataset_train_damage = TrainDataset(
        args.list_train_damage, args, batch_per_gpu=args.batch_size_per_gpu)
    args.root_dataset = args.root_dataset_part
    dataset_train_part = TrainDataset(
        args.list_train_part, args, batch_per_gpu=args.batch_size_per_gpu)
    args.root_dataset = ''

    loader_train_damage = torchdata.DataLoader(
        dataset_train_damage,
        batch_size=args.num_gpus,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)
    loader_train_part = torchdata.DataLoader(
        dataset_train_part,
        batch_size=args.num_gpus,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # create loader iterator
    iterator_train_damage = iter(loader_train_damage)
    iterator_train_part = iter(loader_train_part)

    # load nets into gpu
    if args.num_gpus > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=range(args.num_gpus))
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder_damage, net_decoder_part, crit)
    if args.part_finetune_only:
        freeze_model_except_part(nets, args)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'loss_damage': [], 'acc_damage': [], 'loss_part': [], 'acc_part': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, iterator_train_damage, iterator_train_part, optimizers, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args, epoch)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder_damage', default='ppm_bilinear_deepsup',
                        help="architecture of arch_decoder_damage")
    parser.add_argument('--arch_decoder_part', default='c1_bilinear_deepsup',
                        help="architecture of arch_decoder_part")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder_damage', default='',
                        help="weights to finetune net_decoder_damage")
    parser.add_argument('--weights_decoder_part', default='',
                        help="weights to finetune net_decoder_part")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train_damage',
                        default='./data/train.odgt')
    parser.add_argument('--list_train_part',
                        default='./data/train.odgt')
    parser.add_argument('--root_dataset_damage',
                        default='./data/')
    parser.add_argument('--root_dataset_part',
                        default='./data/')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=8, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder_damage', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder_part', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale_damage', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--deep_sup_scale_part', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')
    parser.add_argument('--part_loss_scale', default=0.3, type=float,
                        help='part semseg loss scale, the final loss is computed as: \
                        damage_loss * (1-part_loss_scale) + part_loss * (part_loss_scale)')
    parser.add_argument('--part_finetune_only', action='store_true',
                        help='only finetune part head, while freeze damage head and backbone')

    # Data related arguments
    parser.add_argument('--num_class_damage', default=6, type=int,
                        help='number of classes')
    parser.add_argument('--num_class_part', default=33, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[300,375,450,525,600], nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')
    parser.add_argument('--class_aware_balance', default=False, type=bool,
                        help='if use class aware sampling when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder_damage = args.lr_decoder_damage
    args.running_lr_decoder_part = args.lr_decoder_part

    args.id += '-' + str(args.arch_encoder)
    args.id += '-' + str(args.arch_decoder_damage)
    args.id += '-' + str(args.arch_decoder_part)
    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    args.id += '-LR_encoder' + str(args.lr_encoder)
    args.id += '-LR_decoder_damage' + str(args.lr_decoder_damage)
    args.id += '-LR_decoder_part' + str(args.lr_decoder_part)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-decay' + str(args.weight_decay)
    args.id += '-fixBN' + str(args.fix_bn)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
