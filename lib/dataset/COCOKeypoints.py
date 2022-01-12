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

import logging

import numpy as np
import torch

import pycocotools
from .COCODataset import CocoDataset

import cv2

logger = logging.getLogger(__name__)


class CocoKeypoints(CocoDataset):
    def __init__(self, cfg, dataset, heatmap_generator=None, offset_generator=None, transforms=None):
        super().__init__(cfg, dataset)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints+1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        self.transforms = transforms

        self.ids = [
            img_id
            for img_id in self.ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]

    def __getitem__(self, idx):
        img, anno, image_info = super().__getitem__(idx)

        print("* image_info : ", image_info)

        mask = self.get_mask(anno, image_info)

        print("* img.shape : ", img.shape, ' ', img.dtype )
        print("* mask.shape : ", mask.shape, ' ', mask.dtype)

        np_mask = np.array([mask], np.uint8)
        np_mask = np.transpose(np_mask, (1,2,0))*255
        print("* np_mask.shape : ", np_mask.shape, ' ', np_mask.dtype)

        np_sum = img*np_mask

        cv2.imshow("np_mask", np_mask)
        cv2.imshow("img", img)
        cv2.imshow("np_sum = img*np_mask", np_sum)
        cv2.waitKey(-1)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]
        joints, area = self.get_joints(anno)

        if self.transforms:
            img, mask_list, joints_list, area = self.transforms(
                img, [mask], [joints], area
            )

        heatmap, ignored = self.heatmap_generator(
            joints_list[0], self.sigma, self.center_sigma, self.bg_weight)
        mask = mask_list[0]*ignored

        offset, offset_weight = self.offset_generator(
            joints_list[0], area)

        return img, heatmap, mask, offset, offset_weight

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints_with_center, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            area[i, 0] = self.cal_area_2_torch(
                torch.tensor(joints[i:i+1,:,:]))

            if obj['area'] < 32**2:
                joints[i, -1, 2] = 0
                continue

            joints_sum = np.sum(joints[i, :-1, :2], axis=0)
            num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 1

        return joints, area

    def get_mask(self, anno, img_info):
        m = np.zeros((img_info['height'], img_info['width']))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5
