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

import os
import cv2
#from .COCODataset import CocoDataset

from .ApolloDataset import Apollo3D

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ApolloKeypoints(Apollo3D):
    def __init__(self, cfg, dataset, heatmap_generator=None, offset_generator=None, transforms=None):
        super().__init__(cfg, dataset)
        # print("INIT_APOLLOKEYPOINTS")
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints+1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        self.transforms = transforms

        self.ignore_images_dir = os.path.join(self.root, 'ignore_mask')

        self.car_kpt_idx = [55, 2, 5, 52, 49, 8, 28, 29, 24, 33, 22, 35, 7, 20, 9, 11, 15, 16, 37, 50, 41, 46, 48, 42]

        self.ids = [
            img_id
            for img_id in self.ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]

    def select_keypoint(self, anno):

        for g_idx in range(len(anno)):
            tmp_list = []

            if anno[g_idx]['num_keypoints'] > 0:
                tmp = np.array(anno[g_idx]['keypoints']).reshape([-1,3])

                for idx in range(len(tmp)):
                    if idx in self.car_kpt_idx:
                        tmp_list.append(tmp[idx][0])
                        tmp_list.append(tmp[idx][1])
                        tmp_list.append(tmp[idx][2])
                    else:
                        if tmp[idx][2] != 0:
                            anno[g_idx]['num_keypoints'] -= 1
                print("tmp_list length : ", len(tmp_list))
                anno[g_idx]['keypoints'] = tmp_list



        return anno

    def __getitem__(self, idx):
        img, anno, image_info = super().__getitem__(idx)

        print(img.shape)

        h,w,c = img.shape

        clip_img = img.copy()
        clip_img = clip_img[h//2-h//4:h//2+h//4, w//2-w//4:w//2+w//4, :]

        clip_resize = cv2.resize(clip_img, (640,640))
        clip_gray = cv2.cvtColor(clip_resize, cv2.COLOR_BGR2GRAY)

        clip_sobel_x = cv2.Sobel(clip_gray, cv2.CV_64F, 1,0, 3)
        clip_sobel_x = cv2.convertScaleAbs(clip_sobel_x)

        clip_sobel_y = cv2.Sobel(clip_gray, cv2.CV_64F, 0,1, 3)
        clip_sobel_y = cv2.convertScaleAbs(clip_sobel_y)

        clip_sobel = cv2.addWeighted(clip_sobel_x, 1, clip_sobel_y,1,0)


        resize_test_img = cv2.resize(img, (640,640))

        resize_gray = cv2.cvtColor(resize_test_img, cv2.COLOR_BGR2GRAY)

        resize_sobel_x = cv2.Sobel(resize_gray, cv2.CV_64F, 1,0, 3)
        resize_sobel_x = cv2.convertScaleAbs(resize_sobel_x)

        resize_sobel_y = cv2.Sobel(resize_gray, cv2.CV_64F, 0,1, 3)
        resize_sobel_y = cv2.convertScaleAbs(resize_sobel_y)

        resize_sobel = cv2.addWeighted(resize_sobel_x, 1, clip_sobel_y,1,0)

        resize_img = cv2.resize(img, (160,160))

        print("image_info : ", image_info)

        # cv2.imshow("origin_resize_img", resize_test_img)
        # cv2.imshow("origin_resize_sobel_img", resize_sobel)
        # cv2.imshow("clip_resize_img", clip_resize)
        # cv2.imshow("clip_sobel_img", clip_sobel)

        mask = self.get_mask(anno, image_info)

        anno = self.select_keypoint(anno)

        anno = [
            obj for obj in anno if obj['num_keypoints'] > 0
        ]

        joints, area = self.get_joints(anno)

        if self.transforms:
            img, mask_list, joints_list, area = self.transforms(
                img, [mask], [joints], area
            )

        print("img.shape : ", img.shape)
        numpy_img = img.permute(1,2,0).mul(255).clamp(0,255).byte().numpy()
        resize_img = cv2.resize(numpy_img, (160,160))

        # print("flip_image.shape : ", numpy_img.shape)

        # cv2.imshow("numpy_img", numpy_img)

        #cv2.waitKey(-1)

        heatmap, ignored = self.heatmap_generator(
            joints_list[0], self.sigma, self.center_sigma, self.bg_weight)

        # print("heatmap.shape : ", heatmap.shape)
        # print("ignored.shape : ", ignored.shape)

        heatmap_vis = heatmap.copy()



        #heatmap_vis = np.transpose(heatmap_vis, (1,2,0))

        print("heatmap_vis.shape : ", heatmap_vis.shape)

        image_fused = np.zeros((160,160,3), dtype=np.uint8)

        for j in range(len(heatmap_vis)):
            # cv2.imshow()
            heatmap_test = heatmap_vis[j]
            # heatmap_test = (heatmap_test > 0.5)
            heatmap_test = heatmap_test*255.0
            # heatmap_test = heatmap_test[heatmap_test>0.5] = 1.0
            # heatmap_test = heatmap_test[heatmap_test<=0.5] = 0.0
            print(np.max(heatmap))
            # print(heat)
            heatmap_test = np.clip(heatmap_test,0,255).astype(np.uint8)

            colored_heatmap = cv2.applyColorMap(heatmap_test, cv2.COLORMAP_JET)

            image_fused[:,:,:] = colored_heatmap*0.7 + resize_img*0.3
            # cv2.imshow("resize_img", resize_img*heatmap_vis[j].reshape(160,160,1))
            # if j == 6:
            #     cv2.imshow("heatmap", image_fused)
            #     cv2.waitKey(-1)
            # plt.imshow(heatmap_vis[j])

        mask = mask_list[0]*ignored

        offset, offset_weight = self.offset_generator(
            joints_list[0], area)

        print("offset", offset)

        print("img.shape : ", img.shape)

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
        mask_name = os.path.join(self.ignore_images_dir, img_info['file_name'])

        m = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION )
        m = cv2.bitwise_not(m)

        return m > 255*0.5
