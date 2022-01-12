# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import HighResolutionModule
from .conv_block import BasicBlock, Bottleneck, AdaptBlock, SEBasicBlock, SEBottleneck

import numpy as np
import cv2

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'SE_BASIC' : SEBasicBlock,
    'BOTTLENECK': Bottleneck,
    'SE_BOTTLENECK' : SEBottleneck,
    'ADAPTIVE': AdaptBlock
}

class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        #up2 = self.up2(low3)
        up2 = nn.Upsample(size=(up1.shape[2], up1.shape[3]), mode='bilinear')(low3)

        return up1 + up2

class Residual(nn.Module):
	def __init__(self, numIn, numOut):
		super(Residual, self).__init__()
		self.numIn = numIn
		self.numOut = numOut
		self.bn = nn.BatchNorm2d(self.numIn)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias = True, kernel_size = 1)
		self.bn1 = nn.BatchNorm2d(self.numOut // 2)
		self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias = True, kernel_size = 3, stride = 1, padding = 1)
		self.bn2 = nn.BatchNorm2d(self.numOut // 2)
		self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias = True, kernel_size = 1)

		if self.numIn != self.numOut:
			self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias = True, kernel_size = 1)

	def forward(self, x):
		residual = x
		out = self.bn(x)
		out = self.relu(out)
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)

		if self.numIn != self.numOut:
			residual = self.conv4(x)

		return out + residual

class PoseHigherResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.SPEC
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i+1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i+2), stage)

        # build head net
        inp_channels = int(sum(self.stages_spec.NUM_CHANNELS[-1]))
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET

        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints+1
        self.offset_prekpt = config_offset['NUM_CHANNELS_PERKPT']

        offset_channels = self.num_joints*self.offset_prekpt
        self.transition_heatmap = self._make_transition_for_head(
                    inp_channels, config_heatmap['NUM_CHANNELS'])
        self.transition_offset = self._make_transition_for_head(
                    inp_channels, offset_channels)
        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(config_offset)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

        # self.deconv1 = nn.ConvTranspose2d(720, 720//2, kernel_size = 5, stride=2, padding=2, output_padding=1, bias=False)
        # self.bn_dc_1 = nn.BatchNorm2d(720//2)
        # self.af_conv = nn.Conv2d(720//2, 720, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn_dc_2 = nn.BatchNorm2d(720)

        # self.deconv1 = nn.ConvTranspose2d(self.nOutChannels, self.nOutChannels//2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)


        # self.nStack = 3
        # self.nModules = 2
        # self.nOutChannels = 64
        # self.nFeats = 720
        #
        # _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_ = [], [], [], [], [], []
        #
        # for i in range(self.nStack):
        #     _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
        #     for j in range(self.nModules):
        #         _Residual.append(Residual(self.nFeats, self.nFeats))
        #     lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1),
        #                                             nn.BatchNorm2d(self.nFeats), self.relu)
        #     _lin_.append(lin)
        #     _tmpOut.append(nn.Conv2d(self.nFeats, self.nOutChannels, bias = True, kernel_size = 1, stride = 1))
        #     _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
        #     _tmpOut_.append(nn.Conv2d(self.nOutChannels, self.nFeats, bias = True, kernel_size = 1, stride = 1))
        #
        # self.hourglass = nn.ModuleList(_hourglass)
        # self.Residual = nn.ModuleList(_Residual)
        # self.lin_ = nn.ModuleList(_lin_)
        # self.tmpOut = nn.ModuleList(_tmpOut)
        # self.ll_ = nn.ModuleList(_ll_)
        # self.tmpOut_ = nn.ModuleList(_tmpOut_)

        # self.deconv1 = nn.ConvTranspose2d(self.nOutChannels, self.nOutChannels//2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        # self.hg_bn2 = nn.BatchNorm2d(self.nOutChannels//2)
        # self.deconv2 = nn.ConvTranspose2d(self.nOutChannels//2, self.nOutChannels//4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        # self.hg_bn3 = nn.BatchNorm2d(self.nOutChannels//4)
        # self.conv2 = nn.Conv2d(self.nOutChannels//4, 1, kernel_size=5, stride=1, padding=2, bias=False)

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)

        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'],
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes,
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                     multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        print("input.shape : ", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("x.shape : ", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print("x.shape : ", x.shape)
        x = self.layer1(x)

        #print("x.shape : ", x.shape)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i+1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i+2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        # print(x0_h)
        # print(x0_w)
        #
        # print(y_list[0].shape)
        # print(y_list[1].shape)
        # print(y_list[2].shape)
        # print(y_list[3].shape)

        x = torch.cat([y_list[0], \
            F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear'), \
            F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')], 1)

        # x_vis = x.clone()
        #
        # x_vis = x_vis.mul(255)\
        #              .clamp(0, 255)\
        #              .byte()\
        #              .cpu().numpy()

        # self.deconv1 = nn.ConvTranspose2d(720, 720//2, kernel_size = 5, stride=2, padding=2, output_padding=1, bias=False)
        # self.bn_dc_1 = nn.BatchNorm2d(720//2)
        # self.af_conv = nn.Conv2d(720//2, 720//4, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn_dc_2 = nn.BatchNorm2d(720//4)

        # x = self.deconv1(x)
        # # print("test x.shape : ", x.shape)
        # x = self.relu(self.bn_dc_1(x))
        # # print("test x.shape : ", x.shape)
        # x = self.af_conv(x)
        # # print("test x.shape : ", x.shape)
        # x = self.relu(self.bn_dc_2(x))
        # # print("test x.shape : ", x.shape)


        # print("test x.shape : ", x.shape)

        final_offset = []
        offset_feature = self.transition_offset(x)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        offset_feature[:,j*self.offset_prekpt:(j+1)*self.offset_prekpt])))

        offset = torch.cat(final_offset, dim=1)

        # print("offset.shape : ",offset.shape)

        # heatmap = []
        #
        #  # stacked hourglass module
        # for i in range(self.nStack):
        #     hg = self.hourglass[i](x)
        #     ll = hg
        #     for j in range(self.nModules):
        #         ll = self.Residual[i * self.nModules + j](ll)
        #     ll = self.lin_[i](ll)
        #     tmpOut = self.tmpOut[i](ll)
        #     heatmap.append(tmpOut)
        #
        #     ll_ = self.ll_[i](ll)
        #     tmpOut_ = self.tmpOut_[i](tmpOut)
        #     x = x + ll_ + tmpOut_
        #     # print("HG Shape : {}".format(x_7.shape))
        #
        # shareFeat = heatmap[-1]
        # print("shareFeat.shape : ", shareFeat.shape)

        # heatmap = self.transition_heatmap(x)
        # print("heatmap.shape 1 : ", heatmap.shape)
        # heatmap = self.head_heatmap[0](heatmap)
        # print("heatmap.shape 2 : ", heatmap.shape)
        # heatmap = self.head_heatmap[1](heatmap)
        # print("heatmap.shape 3 : ", heatmap.shape)

        heatmap = self.head_heatmap[1](
            self.head_heatmap[0](self.transition_heatmap(x)))


        # heatmap = F.upsample(heatmap, size=(320, 320), mode='bilinear')

        # print("heatmap.shape : ", heatmap.shape)

        # heatmap = []
        return heatmap, offset

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained,
                            map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
