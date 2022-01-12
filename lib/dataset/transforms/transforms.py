# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, joints, area):
        for t in self.transforms:
            image, mask, joints, area = t(image, mask, joints, area)
        return image, mask, joints, area

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, mask, joints, area):
        return F.to_tensor(image), mask, joints, area


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints, area):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints, area

class RandomCrop(object):
    def __init__(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def __call__(self, image, mask, joints, area):
        height, width, channel = image.shape

        crop_h = int(height*self.crop_ratio)
        crop_w = int(width*self.crop_ratio)

        # create empty crop image matrix
        crop_image = np.zeros((crop_h, crop_w, channel), dtype=np.uint8)
        # create empty mask image matrix
        crop_mask_image = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # randomly select the center of crop image
        # rand_center_y = np.random.randint(crop_h-crop_h//4, height-crop_h//2)
        rand_center_y = height
        rand_center_x = np.random.randint(crop_w-crop_w//2, crop_w+crop_w//2)

        start_y = rand_center_y - crop_h
        end_y = height
        #end_y = rand_center_y + crop_h//2+1

        # print(mask[0].dtype)

        start_x = rand_center_x - crop_w//2
        end_x = rand_center_x + crop_w//2

        crop_image[:,:,:] = image[start_y:end_y, start_x:end_x, :]
        tmp_mask = (mask[0]).astype(np.uint8)
        crop_mask_image[:,:] = tmp_mask[start_y:end_y, start_x:end_x]

        list_mask = [crop_mask_image.astype(bool)]

        # joint
        for idx in range(len(joints[0])):
            #print(joints[0][idx])
            for kpt_idx in range(len(joints[0][idx])):
                joints[0][idx][kpt_idx][0] = joints[0][idx][kpt_idx][0] - start_x
                joints[0][idx][kpt_idx][1] = joints[0][idx][kpt_idx][1] - start_y
                
                if (joints[0][idx][kpt_idx][0] < 0) or (joints[0][idx][kpt_idx][1] < 0) or (joints[0][idx][kpt_idx][0] > end_x) or (joints[0][idx][kpt_idx][1] > end_y):
                    joints[0][idx][kpt_idx][0] = 0
                    joints[0][idx][kpt_idx][1] = 0
                    joints[0][idx][kpt_idx][2] = 0

        return image, list_mask, joints, area


class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

    def __call__(self, image, mask, joints, area):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)

            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1] - np.zeros_like(mask[i])
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1

        return image, mask, joints, area


class RandomAffineTransform(object):
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate):
        self.input_size = input_size
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate

    def _get_affine_matrix(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        scale = t[0,0]*t[1,1]
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t, scale

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        print(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def __call__(self, image, mask, joints, area):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        height, width = image.shape[:2]

        center = np.array((width/2, height/2))
        if self.scale_type == 'long':
            scale = max(height, width)/200
            print("###################please modify range")
        elif self.scale_type == 'short':
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            center[0] += dx
            center[1] += dy

        for i, _output_size in enumerate(self.output_size):
            mat_output, _ = self._get_affine_matrix(
                center, scale, (_output_size, _output_size), aug_rot
            )
            mat_output = mat_output[:2]
            mask[i] = cv2.warpAffine(
                (mask[i]*255).astype(np.uint8), mat_output,
                (_output_size, _output_size)
            ) / 255
            mask[i] = (mask[i] > 0.5).astype(np.float32)

            joints[i][:, :, 0:2] = self._affine_joints(
                joints[i][:, :, 0:2], mat_output
            )

        mat_input, final_scale = self._get_affine_matrix(
            center, scale, (self.input_size, self.input_size), aug_rot
        )
        mat_input = mat_input[:2]
        area = area*final_scale
        image = cv2.warpAffine(
            image, mat_input, (self.input_size, self.input_size)
        )

        print(joints)
        print("area : ", area)

        return image, mask, joints, area
