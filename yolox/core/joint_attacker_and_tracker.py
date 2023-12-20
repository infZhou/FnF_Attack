# This code is modified from /tools/train.py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from collections import defaultdict
from loguru import logger
import copy, cv2
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import numpy as np
from yolox.utils import (
    fuse_model,
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.byte_tracker import joint_stracks, STrack
from yolox.sort_tracker.sort import Sort
from yolox.sort_tracker.sort import associate_detections_to_trackers, convert_x_to_bbox
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker
from yolox.data.data_augment  import preproc
from tqdm import tqdm
from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import matching

# from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

import datetime
import os
import time

class Attack_Scheduler(object):
    def __init__(self, interval, step):
        self.interval = interval  # conduct attack every [interval] frames
        self.step = min(step, interval)
        self.counter = 0
        self.need_attack = 1

    def next(self):

        if self.counter < self.interval - 1:
            self.counter += 1
        else:
            self.counter = 0
            self.need_attack = 0  # each sequence is only attack once (but may not one frame)

    def is_attack(self):  # determine whether current frame needs to be attacked

        if self.interval == 0:
            return False
        if self.need_attack == 0:
            return False
        if self.counter >= self.interval - self.step:
            return True
        else:
            return False

    def attack_counter(self):
        return self.counter

    def reset(self):
        self.counter = 0
        self.need_attack = 1

def write_results(filename, results, info_imgs, img_size):  # for ByteTrack
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    org_img_w, org_img_h = info_imgs[1].item(), info_imgs[0].item()
    scale = min(img_size[1] / org_img_w, img_size[0] / org_img_h)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                # x1, y1, w, h = tlwh  # disabled by F&F attack
                x1, y1, w, h = tlwh / scale  # added by F&F attack, we fix the input resolution to [1440, 800], we need to scale it back when writing txt files (for evaluating HOTA metrics)

                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_results_no_score(filename, results, info_imgs, img_size):  # for SORT
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    org_img_w, org_img_h = info_imgs[1].item(), info_imgs[0].item()
    scale = min(img_size[1] / org_img_w, img_size[0] / org_img_h)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                # x1, y1, w, h = tlwh  # disabled by F&F attack
                if isinstance(tlwh, list):
                    tlwh = np.array(tlwh)
                x1, y1, w, h = tlwh / scale  # added by F&F attack, we fix the input resolution to [1440, 800], we need to scale it back when writing txt files (for evaluating HOTA metrics)

                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

class Joint_Attacker_and_Tracker:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = 1
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = False  # disabled by F&F attack

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(self.file_name, distributed_rank=self.rank, filename="attack_log.txt", mode="a")

        # adv settings
        self.attack_scheduler = Attack_Scheduler(args.attack_interval, args.attack_step_per_interval)
        self.attack_scheduler.reset()
        self.loss_stats = ['loss', 'conf_loss', 'bbox_loss']


        self.debug_dir = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'exp', self.args.experiment_name, 'debug')
        os.makedirs(self.results_folder, exist_ok=True)

        self.results_folder = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'pred_txt_results', args.experiment_name)
        os.makedirs(self.results_folder, exist_ok=True)

        self.crit_diff_item = ['conf_loss', 'cls_loss', 'iou_loss', 'l1_loss',
                               'num_det', 'adv_loss']  # record the changes of various items during attack

        self.crit_key = 'adv_loss'

        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.num_classes = exp.num_classes
        self.img_size = exp.test_size
        self.rgb_means = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        self.std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]

        self.clip_low = (0 - self.rgb_means) / self.std  # clip bound used to ensure the perturbed input is within [0, 1]
        self.clip_high = (1 - self.rgb_means) / self.std


    def attack_and_track(self):
        self.before_attack()
        try:
            self.joint_attack_and_track_in_epoch()
        except Exception:
            raise

    def joint_attack_and_track_in_epoch(self):
        for self.epoch in range(1):
            self.before_epoch()
            self.joint_attack_and_track_in_iter()
            # self.after_epoch()

    def get_target_gt(self, targets):

        GAMMA = 4
        KAPPA = self.args.fp_shift_ratio  # default: 0.2
        ct_x_shift = [KAPPA, KAPPA, -KAPPA, -KAPPA]
        ct_y_shift = [KAPPA, -KAPPA, KAPPA, -KAPPA]
        targets_adv = torch.zeros_like(targets)  # shape [1, num_gt, 6]
        num_target = (targets[0, :, 5] > 0).sum()
        for k in range(num_target):
            for ifp in range(GAMMA):
                new_k = k * GAMMA + ifp
                targets_adv[0, new_k] = targets[0, k] * 1.0
                targets_adv[0, new_k, 1] += targets_adv[0, new_k, 3] * ct_x_shift[ifp]
                targets_adv[0, new_k, 2] += targets_adv[0, new_k, 4] * ct_y_shift[ifp]
                targets_adv[0, new_k, [3,4]] *= self.args.fp_scale_ratio
        return targets_adv


    def joint_attack_and_track_in_iter(self):
        tracker = None
        NUM_IDSW_DICT, COMMON_GT_ID_SET = {}, {}
        results = []
        sub_seq_count = 0
        last_main_seq_name = None
        video_names = defaultdict()
        main_seq_name_bit = 14 if not self.args.mot20 else 8


        for self.sample_idx, (inps, targets, info_imgs, ids) in enumerate(tqdm(self.adv_loader)):

            inps = inps.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if self.attack_scheduler.is_attack():
                targets_adv = self.get_target_gt(targets.clone())
            else:
                targets_adv = targets.clone()

            # conduct or skip attack according to self.attack_scheduler
            if self.attack_scheduler.is_attack():
                self.model.adversarial = True
                self.model.head.adversarial = True

                img_to_forward = self.attack_one_sample(inps, targets_adv, info_imgs, ids, return_adv_img=True).unsqueeze(0)

                self.model.adversarial = False
                self.model.head.adversarial = False
            else:
                self.model.adversarial = False
                self.model.head.adversarial = False
                img_to_forward = inps

            info_imgs_org = copy.deepcopy(info_imgs)

            info_imgs[0][0] = 800  # overwrite height
            info_imgs[1][0] = 1440  # overwrite width



            # conduct inference with clean/attacked frame
            tensor_type = torch.cuda.HalfTensor if self.args.fp16 else torch.cuda.FloatTensor
            self.model = self.model.eval()
            if self.args.fp16:
                self.model = self.model.half()

            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name not in NUM_IDSW_DICT:
                    NUM_IDSW_DICT[video_name] = {'overall':0}
                    COMMON_GT_ID_SET[video_name] = {size_key: [] for size_key in ['overall']}

                if not self.args.enable_sort_tracker:  # use BYTETrack
                    if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                        self.args.track_buffer = 14
                    elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                        self.args.track_buffer = 25
                    else:
                        self.args.track_buffer = 30

                    if video_name == 'MOT17-01-FRCNN':
                        self.args.track_thresh = 0.65
                    elif video_name == 'MOT17-06-FRCNN':
                        self.args.track_thresh = 0.65
                    elif video_name == 'MOT17-12-FRCNN':
                        self.args.track_thresh = 0.7
                    elif video_name == 'MOT17-14-FRCNN':
                        self.args.track_thresh = 0.67
                    elif video_name in ['MOT20-06', 'MOT20-08']:
                        self.args.track_thresh = 0.3
                    else:
                        self.args.track_thresh = self.args.track_thresh

                if frame_id == 1:
                    if self.args.n_frames_per_seq > 0:
                        if video_name[:main_seq_name_bit]!=last_main_seq_name and last_main_seq_name is not None:
                            sub_seq_count = 0
                        video_name = video_name + '-' + str(sub_seq_count).zfill(3)
                        sub_seq_count += 1
                        last_main_seq_name = video_name[:main_seq_name_bit]
                    if video_name not in video_names:
                        video_names[video_id] = video_name

                    if not self.args.enable_sort_tracker:  # use BYTETrack
                        tracker = BYTETracker(self.args)
                    else:  # use SORT
                        tracker = Sort(self.args, det_thresh=0.4, min_hits=0 if self.args.disable_strict_birth else 1)

                    self.attack_scheduler.reset()

                img_to_forward = img_to_forward.type(tensor_type)

                outputs = self.model(img_to_forward)

                outputs_det_debug = postprocess(outputs.clone(), self.num_classes, 0.7, self.nmsthre)  # for visualization

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    if not self.args.enable_sort_tracker:  # for BYTETrack
                        tlwh = t.tlwh
                        tid = t.track_id
                    else:  # for SORT
                        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                        tid = t[4]

                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        if not self.args.enable_sort_tracker:
                            online_scores.append(t.score)
                # save results
                if not self.args.enable_sort_tracker:
                    results.append((frame_id, online_tlwhs, online_ids, online_scores))
                else:
                    results.append((frame_id, online_tlwhs, online_ids))

            if frame_id == self.args.n_frames_per_seq:
                if len(results) != 0:
                    result_filename = os.path.join(self.results_folder, '{}.txt'.format(video_names[video_id]))
                    if not self.args.enable_sort_tracker:
                        write_results(result_filename, results, info_imgs_org, self.img_size)
                    else:
                        write_results_no_score(result_filename, results, info_imgs_org, self.img_size)
                    results = []

            # visualization
            if self.args.debug in [6]:
                org_img_h, org_img_w = info_imgs[0].item(), info_imgs[1].item()
                img0 = img_to_forward[0].detach().cpu().permute(1, 2, 0).numpy()
                img0 = (img0 * self.std + self.rgb_means) * 255
                img0 = np.clip(img0, 0, 255).astype(np.uint8)
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)


                im_plot_track = np.ascontiguousarray(np.copy(img0))
                im_plot_det = np.ascontiguousarray(np.copy(img0))
                im_plot_gt = np.ascontiguousarray(np.copy(img0))
                im_h, im_w = im_plot_track.shape[:2]

                text_scale = max(1, img0.shape[1] / 1600.)
                text_thickness = 1 if text_scale > 1.1 else 1
                line_thickness = max(1, int(img0.shape[1] / 500.))
                line_thickness = 3 # added by F&F attack

                # plot tracklets
                scale = min(im_w / org_img_w, im_h / org_img_h)
                for i_track in range(len(online_tlwhs)):
                    x1, y1, w, h = online_tlwhs[i_track]
                    track_id = online_ids[i_track]
                    x1, y1, w, h = x1 * scale, y1 * scale, w * scale, h * scale
                    intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                    _line_thickness = line_thickness
                    color = (int(track_id * 71 % 255), int(track_id * 41 % 255), int(track_id * 37 % 255))
                    cv2.rectangle(im_plot_track, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)

                # plot dets
                dets = self.convert_to_coco_format(outputs_det_debug, info_imgs, ids)
                # print(f'num_det={len(dets)}')
                for det in dets:
                    x1, y1, w, h = det['bbox']
                    x1, y1, w, h = x1 * scale, y1 * scale, w * scale, h * scale
                    intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                    _line_thickness = line_thickness
                    cv2.rectangle(im_plot_det, intbox[0:2], intbox[2:4], color=(255, 255, 255),
                                  thickness=line_thickness)

                # plot gts
                gt_bboxes = targets_adv[0][targets_adv[0][:,-1] > 0, 1:5].cpu().numpy()
                for gt_bbox in gt_bboxes:
                    x1, y1, w, h = gt_bbox
                    x1, y1, w, h = x1 * scale, y1 * scale, w * scale, h * scale
                    intbox = tuple(map(int, (x1-w*0.5, y1-h*0.5, x1 + w*0.5, y1 + h*0.5)))
                    _line_thickness = line_thickness
                    cv2.rectangle(im_plot_gt, intbox[0:2], intbox[2:4], color=(255, 255, 255),
                                  thickness=line_thickness)

                # save
                save_dir = os.path.join(self.debug_dir, 'generic')
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_dir + '/{}_frame_{}.jpg'.format(info_imgs[-1][0][:8], self.sample_idx), im_plot_track)

                save_dir = os.path.join(self.debug_dir, 'detection')
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_dir + '/{}_frame_{}.jpg'.format(info_imgs[-1][0][:8], self.sample_idx), im_plot_det)

                save_dir = os.path.join(self.debug_dir, 'gt_bbox')
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_dir + '/{}_frame_{}.jpg'.format(info_imgs[-1][0][:8], self.sample_idx), im_plot_gt)


            if frame_id == 1:  # record initial tracking identities
                valid_gt_mask = targets[0][:,-1] > 0
                gt_bboxes = targets[0][valid_gt_mask, 1:5].cpu().numpy()
                gt_ids = targets[0][valid_gt_mask, 5].cpu().numpy()
                gt_sizes = gt_bboxes[:, 2] * gt_bboxes[:, 3]
                pred_ids = np.array(online_ids)
                pred_bboxes = np.stack(online_tlwhs) if len(online_tlwhs) > 0 else np.zeros((0,4))
                gt_bboxes[:, [0, 1]] = gt_bboxes[:, [0, 1]] - gt_bboxes[:, [2, 3]] * 0.5
                gt_bboxes[:, [2, 3]] = gt_bboxes[:, [0, 1]] + gt_bboxes[:, [2, 3]]
                pred_bboxes[:, [2, 3]] = pred_bboxes[:, [0, 1]] + pred_bboxes[:, [2, 3]]
                gt_id_2_pred_id_before_atk = {}
                gt_id_set_before_atk = {'overall':[]}

                iou_mtx = 1 - bbox_ious(np.ascontiguousarray(gt_bboxes, dtype=np.float),
                                        np.ascontiguousarray(pred_bboxes, dtype=np.float))
                matches, u_gt_ind, u_pred_ind = matching.linear_assignment(iou_mtx, thresh=0.5)

                for i_m in range(matches.shape[0]):
                    gt_id = gt_ids[matches[i_m, 0]]
                    pred_id = pred_ids[matches[i_m, 1]]
                    gt_id_2_pred_id_before_atk[gt_id] = pred_id
                    gt_id_set_before_atk['overall'].append(gt_id)

            # get tracking identities after attack and count immediate IDSW
            if frame_id == self.args.attack_interval+3:  # we leave several additional clean frames waiting for the Kalman filter
                valid_gt_mask = targets[0][:,-1] > 0
                gt_bboxes = targets[0][valid_gt_mask, 1:5].cpu().numpy()
                gt_ids = targets[0][valid_gt_mask, 5].cpu().numpy()
                gt_sizes = gt_bboxes[:, 2] * gt_bboxes[:, 3]
                pred_ids = np.array(online_ids)
                pred_bboxes = np.stack(online_tlwhs) if len(online_tlwhs) > 0 else np.zeros((0,4))
                gt_bboxes[:, [0, 1]] = gt_bboxes[:, [0, 1]] - gt_bboxes[:, [2, 3]] * 0.5
                gt_bboxes[:, [2, 3]] = gt_bboxes[:, [0, 1]] + gt_bboxes[:, [2, 3]]
                pred_bboxes[:, [2, 3]] = pred_bboxes[:, [0, 1]] + pred_bboxes[:, [2, 3]]
                gt_id_2_pred_id_after_atk = {}
                gt_id_set_after_atk = {'overall':[]}

                iou_mtx = 1 - bbox_ious(np.ascontiguousarray(gt_bboxes, dtype=np.float),
                                        np.ascontiguousarray(pred_bboxes, dtype=np.float))
                matches, u_gt_ind, u_pred_ind = matching.linear_assignment(iou_mtx, thresh=0.5)

                for i_m in range(matches.shape[0]):
                    gt_id = gt_ids[matches[i_m, 0]]
                    pred_id = pred_ids[matches[i_m, 1]]
                    gt_id_2_pred_id_after_atk[gt_id] = pred_id
                    gt_id_set_after_atk['overall'].append(gt_id)

                common_gt_id_set = {size_key: list(set(gt_id_set_before_atk[size_key]).intersection(set(gt_id_set_after_atk[size_key]))) \
                                    for size_key in ['overall']
                                    }
                num_IDSW = {'overall':0}
                for gt_id in common_gt_id_set['overall']:
                    if gt_id_2_pred_id_before_atk[gt_id] != gt_id_2_pred_id_after_atk[gt_id]:
                        num_IDSW['overall'] += 1

                # for size_key in ['overall']:
                #     print('{}: num_IDSW = {}, num_common_gt_id = {}, IDSW_ratio = {:.2f}'.format(
                #         size_key, num_IDSW[size_key], len(common_gt_id_set[size_key]), 100 * num_IDSW[size_key] / len(common_gt_id_set[size_key]) if len(common_gt_id_set[size_key]) > 0 else 0))

                # merge results of each sub sequence (30 frames each)
                main_seq_name = video_name[:main_seq_name_bit]
                for size_key in ['overall']:
                    NUM_IDSW_DICT[main_seq_name][size_key] += num_IDSW[size_key]
                    COMMON_GT_ID_SET[main_seq_name][size_key] += common_gt_id_set[size_key]

                for size_key in ['overall']:
                    logger.info('seq_name:{}, accumulated: num_IDSW = {}, num_common_gt_id = {}, IDSW_ratio (attack success rate) = {:.2f}'.format(
                        main_seq_name, NUM_IDSW_DICT[main_seq_name][size_key], len(COMMON_GT_ID_SET[main_seq_name][size_key]),
                        100 * NUM_IDSW_DICT[main_seq_name][size_key] / len(COMMON_GT_ID_SET[main_seq_name][size_key]) if len(COMMON_GT_ID_SET[main_seq_name][size_key]) > 0 else 0))

            self.attack_scheduler.next()

        logger.info('='*80)
        logger.info('Exp Name: ', self.args.experiment_name)
        logger.info('Attack success rate per sequence:')
        for main_seq_name in NUM_IDSW_DICT:
            logger.info(main_seq_name)
            for size_key in ['overall']:
                logger.info('{}: num_IDSW = {}, num_common_gt_id = {}, IDSW_ratio = {:.2f}'.format(
                    size_key, NUM_IDSW_DICT[main_seq_name][size_key], len(COMMON_GT_ID_SET[main_seq_name][size_key]),
                    100 * NUM_IDSW_DICT[main_seq_name][size_key] / len(COMMON_GT_ID_SET[main_seq_name][size_key]) if len(
                        COMMON_GT_ID_SET[main_seq_name][size_key]) > 0 else 0))
        logger.info('=' * 80)
        logger.info('Overall attack success rate')
        for size_key in ['overall']:
            overall_idsw = 0
            overall_n_gt = 0
            for m_seq_name in NUM_IDSW_DICT:
                overall_idsw += NUM_IDSW_DICT[m_seq_name][size_key]
                overall_n_gt += len(COMMON_GT_ID_SET[m_seq_name][size_key])

            logger.info('{}: num_IDSW = {}, num_common_gt_id = {}, IDSW_ratio = {:.2f}'.format(
                size_key, overall_idsw, overall_n_gt,
                100 * overall_idsw / overall_n_gt if overall_n_gt > 0 else 0))

    def attack_one_sample(self, inps, targets, info_imgs, ids, return_adv_img=False):

        iter_start_time = time.time()
        track_ids = targets[:, :, 5]
        targets = targets[:, :, :5]
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        # init attack param dictionary
        adv_param_dict = self.init_adv_param(inps)


        inps_adv = inps.clone()
        targets_adv = targets.clone()


        adv_img_dict = {}
        self.best_crit = -1e10

        for self.iter in range(self.args.max_attack_iter+2):  # the perturbation actually starts to become effective since self.iter==2
            if self.iter > 0:
                adv_perturbation = self.render_perturbation(adv_param_dict)
                inps_adv[0] = inps[0] + adv_perturbation

                for cha in range(3):  # clip the perturbed input to be within [0, 1]
                    inps_adv[0, cha] = torch.clamp(inps_adv[0, cha].clone(),
                                                   min=self.clip_low[0, 0, cha],
                                                   max=self.clip_high[0, 0, cha])

                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    loss, preds = self.model(inps_adv, targets_adv, self.iter)
            else:
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    loss, preds = self.model(inps_adv, targets_adv, self.iter)

            if self.args.attack_mode in [6]:
                adv_loss = - loss['conf_loss'] * self.args.adv_conf_loss_weight \
                           - loss['cls_loss'] * self.args.adv_cls_loss_weight \
                           - loss['iou_loss'] * self.args.adv_iou_loss_weight \
                           - loss['l1_loss'] * self.args.adv_l1_loss_weight

            adv_loss = adv_loss.sum()

            if self.iter > 0:
                adv_param_dict = self.update_adv_param(adv_loss, adv_param_dict)

            b_id = 0  # short for "batch_size_id", used for parallel decode, current version only support batch_size = 1

            is_best = adv_loss.item() > self.best_crit

            if self.iter > 0  and is_best:
                adv_img_dict[b_id] = inps_adv[b_id].detach().clone()
                self.best_crit = adv_loss.item()

            if self.iter == self.args.max_attack_iter+1 and self.best_crit == 0:  # this means there is few objects in the clean frame
                adv_img_dict[b_id] = inps_adv[b_id].detach().clone()

            # print('atk_iter={}, best_crit={}'.format(self.iter, self.best_crit))

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=0,
            **loss,
        )

        if return_adv_img:
            return adv_img_dict[0]

    def before_attack(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model(adversarial=True, attack_mode=self.args.attack_mode)
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.fnf_load_model(model)
        model.eval()


        # data related init
        self.no_aug = True
        self.adv_loader = self.exp.get_adv_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
        )
        # len_data_loader means iters per epoch
        self.len_data_loader = len(self.adv_loader)


        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.len_data_loader * self.start_epoch

        self.model = model

        # Tensorboard logger
        # added by F&F attack ---------------------------------
        USE_TENSORBOARD = True
        try:
            import tensorboardX
        except:
            USE_TENSORBOARD = False
        if self.rank == 0 and USE_TENSORBOARD:
            self.tblogger = tensorboardX.SummaryWriter(self.file_name)
        # --------------------------------------------

        logger.info("Attacking start...")


    def before_epoch(self):

        if self.no_aug:
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.len_data_loader * self.args.max_attack_iter - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "img: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.len_data_loader, self.iter + 1, self.args.max_attack_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        # if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
        #     self.input_size = self.exp.random_resize(
        #         self.adv_loader, self.epoch, self.rank, self.is_distributed
        #     )

    def init_adv_param(self, inps):
        '''

        [input]
        inps: input image, shape = torch.Size([1, 3, 800, 1440])
        '''

        universal_rgb = None

        delta = torch.zeros_like(inps).cuda()
        delta.requires_grad = True

        return {'delta': delta, 'universal_rgb': universal_rgb}

    def render_perturbation(self, adv_param_dict):
        '''
        render perturbation according to adv_param
        '''

        perturbation = adv_param_dict['delta']

        return perturbation


    def update_adv_param(self, adv_loss, adv_param_dict):
        '''
        Update the perturbation
        '''

        epsilon = self.args.epsilon  / self.std
        epsilon = torch.tensor(epsilon[0]).view((1,3,1,1)).cuda()


        noise_upd_step = self.args.alpha / self.std
        noise_upd_step = torch.tensor(noise_upd_step[0]).view((1, 3, 1, 1)).cuda()

        grad = torch.autograd.grad(adv_loss, [adv_param_dict['delta']])[0].detach()


        delta = noise_upd_step * torch.sign(grad)
        if self.args.adv_grad_clip != 0:
            mask = (torch.abs(grad) > self.args.adv_grad_clip).float()
            delta = mask * delta

        adv_param_dict['delta'] = adv_param_dict['delta'] + delta

        # clip the perturbation to meet the l_\inf constraint
        for cha in range(3):
            adv_param_dict['delta'][0,cha] = torch.clamp(adv_param_dict['delta'][0,cha],
                                                         min=-epsilon[0, cha].item(), max=epsilon[0, cha].item())


        return adv_param_dict

    @property
    def progress_in_iter(self):
        return self.epoch * self.len_data_loader + self.iter

    def fnf_load_model(self, model):
        '''
        implemented by F&F attack, modified from /tools/track.py
        '''
        rank = self.args.local_rank
        results_folder = os.path.join(self.file_name, "track_results")
        os.makedirs(results_folder, exist_ok=True)
        logger.info("Args: {}".format(self.args))

        if self.args.ckpt is not None:
            ckpt_file = self.args.ckpt
        else:
            raise ValueError('[yolox/core/attacker.py] No checkpoint are given!')
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        return model

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.adv_loader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].detach().numpy().tolist(),
                    "score": scores[ind].detach().numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def debug(self, batch, output, iter_id):
        raise NotImplementedError