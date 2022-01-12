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
import numpy as np
import argparse
import os
import sys
import stat
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from dataset import make_test_dataloader
from utils.utils import create_logger
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.rescore import rescore_valid

from dataset.target_generators import HeatmapGenerator

import cv2

heatmap_generator = HeatmapGenerator(
    cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS
)

self_sigma = cfg.DATASET.SIGMA
self_center_sigma = cfg.DATASET.CENTER_SIGMA
self_bg_weight = cfg.DATASET.BG_WEIGHT

torch.multiprocessing.set_sharing_strategy('file_system')

def generate_random_color(color_list, num_keypoints):

    for n_kpt in range(num_keypoints):
        color_list.append(list(np.random.randint(0,255, size=(3,))))

    return color_list

VEHICLE_KEYPOINT_COLOR = []

VEHICLE_KEYPOINT_COLOR = generate_random_color(VEHICLE_KEYPOINT_COLOR, 66)

print(VEHICLE_KEYPOINT_COLOR)

# VEHICLE_KEYPOINT_ACTIVATE_INDEXES = [55,2,5,52,49,8,28,29,24,33,22,35,7,20,9,11,15,16,37,50,41,46,48,42]
VEHICLE_KEYPOINT_ACTIVATE_INDEXES = {
    0: 2,
    1: 5,
    2: 7,
    3: 8,
    4: 9,
    5: 11,
    6: 15,
    7: 16,
    8: 20,
    9: 22,
    10: 24,
    11: 28,
    12: 29,
    13: 33,
    14: 35,
    15: 37,
    16: 41,
    17: 42,
    18: 46,
    19: 48,
    20: 49,
    21: 50,
    22: 52,
    23: 55
}
print(sorted(VEHICLE_KEYPOINT_ACTIVATE_INDEXES))

def make_heatmap(image, heatmaps):
    heatmaps = heatmaps.mul(255)\
                       .clamp(0,255)\
                       .byte()\
                       .cpu().numpy()

    num_joints, height, width = heatmaps.shape

    image_resized = cv2.resize(image, (int(width), int(height)))
    print("image_resized.shape  : ",image_resized.shape)

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        heatmap = heatmaps[j, : , :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width*(j+1)
        width_end = width*(j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    # cv2.imshow('image_grid', image_grid)
    # cv2.waitKey(-1)

    return image_grid

def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h

def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    print("tagmaps.shape : ", tagmaps.shape)
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu()\
                       .numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    # cv2.imshow('image_grid', image_grid)
    # cv2.waitKey(-1)

    return image_grid

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )

def get_joints(anno):
    num_joints = cfg.DATASET.NUM_JOINTS
    num_joints_with_center = num_joints + 1

    num_people = len(anno)
    area = np.zeros((num_people, 1))
    joints = np.zeros((num_people, num_joints_with_center, 3))



    for i, obj in enumerate(anno):

        joints[i, :num_joints, :3] = \
            np.array(obj['keypoints']).reshape([-1, 3])

        area[i, 0] = cal_area_2_torch(
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

def select_keypoint(anno):
    # [5,7,20,28,29,37,50,52]
    # car_kpt_idx = [52,5,7,20,28,29,37,50]
    car_kpt_idx = [55, 2, 5, 52, 49, 8, 28, 29, 24, 33, 22, 35, 7, 20, 9, 11, 15, 16, 37, 50, 41, 46, 48, 42]
    for g_idx in range(len(anno)):
        tmp_list = []

        if anno[g_idx]['num_keypoints'] > 0:
            tmp = np.array(anno[g_idx]['keypoints']).reshape([-1,3])

            for idx in range(len(tmp)):
                if idx in car_kpt_idx:
                    tmp_list.append(tmp[idx][0])
                    tmp_list.append(tmp[idx][1])
                    tmp_list.append(tmp[idx][2])
                else:
                    if tmp[idx][2] != 0:
                        anno[g_idx]['num_keypoints'] -= 1
            #print("tmp_list length : ", len(tmp_list))
            anno[g_idx]['keypoints'] = tmp_list



    return anno

def calc_box_size(vehicle):

    x_array = list()
    y_array = list()

    for visible in vehicle:
        if visible[2] > 0.0:
            x_array.append(visible[0])
            y_array.append(visible[1])

    if len(x_array) == 0 or len(y_array) == 0:
        return -1


    right = max(x_array)
    left = min(x_array)
    up = min(y_array)
    down = max(y_array)

    dist = np.sqrt((up-down)**2 + (right-left)**2)

    return dist

def calc_dist(pred, gt):
    return np.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2)

def calc_pck(pred, gt):
    gt = gt[:, :-1, :]
    # print("gt.shape : ", gt.shape)
    np_pred = np.array(pred)
    # print("pred.shape : ", np_pred.shape)

    n_vehicle = gt.shape[0]

    print(n_vehicle)

    exist_keypoints = np.zeros(gt.shape[1])
    correct_keypoints = np.zeros(gt.shape[1])

    for ve in range(n_vehicle):
        vehicle = gt[ve]

        vehicle_size = calc_box_size(vehicle)

        if vehicle_size == -1:
            break

        # print(vehicle_size)
        # for key_idx, key_pt in enumerate(vehicle):
        #     if key_pt[2] > 0.0:
        #         cmp_gt = key_pt[:2]
        #         exist_keypoints[key_idx] += 1

        for key_idx, key_pt in enumerate(vehicle):
            if key_pt[2] > 0.0:
                cmp_gt = key_pt[:2]
                exist_keypoints[key_idx] += 1

                for pred_test in np_pred:
                    cmp_kpt = pred_test[key_idx][:2]

                    kpt_dist = calc_dist(cmp_kpt, cmp_gt)

                    # if kpt_dist < 0.05*vehicle_size:
                    if kpt_dist < 10.0:
                        correct_keypoints[key_idx] += 1
                        break

    # print(exist_keypoints)
    # print(correct_keypoints)

    return exist_keypoints, correct_keypoints


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _ = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    all_reg_preds = []
    all_reg_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None

    exist = np.zeros(cfg.DATASET.NUM_JOINTS)
    correct = np.zeros(cfg.DATASET.NUM_JOINTS)

    for i, (images, anno) in enumerate(data_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'
        # if i == 10: break
        image = images[0].cpu().numpy()
        print("image.shape : ", image.shape)

        anno = select_keypoint(anno)

        joints, area = get_joints(anno)

        # print("joints.shape : ", joints.shape)

        # vis_image = cv2.resize(image, (160,160))

        # cv2.imshow("image", image)

        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        print("base size : {}".format(base_size))
        print("center : {}".format(center))
        print("scale : {}".format(scale))

        with torch.no_grad():
            heatmap_sum = 0
            poses = []

            heatmap_generator = HeatmapGenerator(
                cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS
            )

            print(joints.shape)
            print(joints)

            heatmap_joints = joints.copy()
            # heatmap_joints[:,:,0] = (heatmap_joints[:,:,0] / 3384) * 160
            # heatmap_joints[:,:,1] = (heatmap_joints[:,:,1] / 2710) * 160

            heatmap_joints[:,:,0] = (heatmap_joints[:,:,0] / 3384.0) * 160
            heatmap_joints[:,:,1] = (heatmap_joints[:,:,1] / 2710.0) * 160

            gt_heatmap, ignored = heatmap_generator(
                heatmap_joints, 1, 2, 0.1)

            print("gt_heatmap.shape" , gt_heatmap.shape)
            print("gt_heatmap.dtype" , gt_heatmap.dtype)

            torch_gt_heatmap = torch.from_numpy(gt_heatmap)
            print("torch_gt_heatmap.shape" , torch_gt_heatmap.shape)



            for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                height, width, channel = image.shape

                crop_h = int(height*0.5)
                crop_w = int(width*0.5)

                crop_image = np.zeros((crop_h, crop_w, channel), dtype=np.uint8)

                crop_image[:,:,:] = image[crop_h:,width//2 - crop_w//2:width//2 + crop_w//2,:]


                image_resized, center, scale_resized = resize_align_multi_scale(
                    image, cfg.DATASET.INPUT_SIZE, scale, 1.0
                )

                image_resized = cv2.resize(crop_image, (640,640))
                vis_image = crop_image.copy()

                #print("image_resized.shape : ", image_resized.shape)

                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                heatmap, posemap = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST
                )

                #print(posemap.shape)

                #print("heatmap.shape : ",heatmap.shape)

                heatmap_sum, poses = aggregate_results(
                    cfg, heatmap_sum, poses, heatmap, posemap, scale
                )
                #print("poses.shape : ", poses[0].shape)
                #print(poses)

                #print("heatmap_sum.shape : ", heatmap_sum.shape)


            heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
            poses, scores = pose_nms(cfg, heatmap_avg, poses)

            print("heatmap.shape : ", heatmap.shape)

            grid_image = make_heatmap(vis_image, heatmap[0])
            gt_grid = make_heatmap(vis_image, torch_gt_heatmap)

            print("gt_grid.shape : ", gt_grid.shape)


            #gt_image = make_heatmap(vis_image, gt_heatmap)
            # vis_image_resized = cv2.resize(vis_image, (160,160))
            #
            # for kq in range(len(gt_heatmap)):
            #     gt_htmap = gt_heatmap[kq, :, :]
            #     colored_heatmap = cv2.applyColorMap(gt_htmap, cv2.COLORMAP_JET)
            #     image_fused = colored_heatmap*0.5 + vis_image_resized*0.5
            #     cv2.imshow("image_fused", image_fused)
            #     cv2.waitKey(-1)
            #
            #     print(image_fused.shape)
            #
            #     width_begin = 160*(kq+1)
            #     width_end = 160*(kq+2)
            #     gt_grid[:, width_begin:width_end, :] = image_fused
            #
            # # gt_grid[:, 0:160, :] = vis_image_resized

            cv2.imshow("grid_image", grid_image)
            cv2.imshow("grid_grid", gt_grid)
            cv2.waitKey(-1)
            pose_grid_image = make_tagmaps(vis_image, posemap[0])

            if len(scores) == 0:
                all_reg_preds.append([])
                all_reg_scores.append([])
            else:
                if cfg.TEST.MATCH_HMP:
                    poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

                final_poses = get_final_preds(
                    poses, center, scale_resized, base_size
                )
                if cfg.RESCORE.VALID:
                    scores = rescore_valid(cfg, final_poses, scores)
                all_reg_preds.append(final_poses)
                all_reg_scores.append(scores)

            final_results = []


            # print("final_poses.shape : ", len(final_poses))#.shape)

            # exist_kpts, correct_kpts = calc_pck(final_poses, joints)
            #
            # exist += exist_kpts
            # correct += correct_kpts
            #
            # print(exist)
            # print(correct)
            #
            # # gt_heatmap = heatmap_generator(
            # #     joints[0], self_sigma, self_center_sigma, self_bg_weight)
            #
            # for j in range(len(scores)):
            #     if scores[j] > 0.1:
            #         final_results.append(final_poses[j])
            #
            #
            #
            #
            # for coords in final_results:
            #     x_list = []
            #     y_list = []
            #     print(final_results)
            #     for coord_idx, coord in enumerate(coords):
            #         x_coord, y_coord = int(coord[0]), int(coord[1])
            #         x_list.append(x_coord)
            #         y_list.append(y_coord)
            #
            #         color = (int(VEHICLE_KEYPOINT_COLOR[coord_idx][0]), int(VEHICLE_KEYPOINT_COLOR[coord_idx][1]), int(VEHICLE_KEYPOINT_COLOR[coord_idx][2]))
            #
            #         cv2.circle(vis_image, (x_coord, y_coord), 4, color, 2)
            #
            #     right = max(x_list)
            #     left = min(x_list)
            #     up = min(y_list)
            #     down = max(y_list)
            #
            #     cv2.rectangle(vis_image, (left, up), (right, down), (0,255,0), 2)
            #
            #
            # cv2.imshow("vis_image", vis_image)
            # # cv2.imwrite("/home/sun/Desktop/DEKR_Vehicle/output/images/"+ str(i)+".png", vis_image)
            # cv2.waitKey(-1)
    #     print(exist)
    #     print(correct)
    #
    # detection_rate = sum(correct) / sum(exist)
    #
    # detection_rate = detection_rate * 100.0
    # print("detection rate : ", detection_rate)











    #     if cfg.TEST.LOG_PROGRESS:
    #         pbar.update()
    #
    # sv_all_preds = [all_reg_preds]
    # sv_all_scores = [all_reg_scores]
    # sv_all_name = [cfg.NAME]
    #
    # if cfg.TEST.LOG_PROGRESS:
    #     pbar.close()
    #
    # for i in range(len(sv_all_preds)):
    #     print('Testing '+sv_all_name[i])
    #     preds = sv_all_preds[i]
    #     scores = sv_all_scores[i]
    #     if cfg.RESCORE.GET_DATA:
    #         test_dataset.evaluate(
    #             cfg, preds, scores, final_output_dir, sv_all_name[i]
    #         )
    #         print('Generating dataset for rescorenet successfully')
    #     else:
    #         name_values, _ = test_dataset.evaluate(
    #             cfg, preds, scores, final_output_dir, sv_all_name[i]
    #         )
    #
    #         if isinstance(name_values, list):
    #             for name_value in name_values:
    #                 _print_name_value(logger, name_value, cfg.MODEL.NAME)
    #         else:
    #             _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
