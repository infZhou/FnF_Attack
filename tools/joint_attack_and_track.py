'''
This code is modified from /tools/train.py
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Joint_Attacker_and_Tracker, launch
from yolox.exp import get_exp

import argparse
import random
import warnings
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size, currently only support 1")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for attacking, currently only support 1")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training, not used in F&F attack")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your expriment description file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true", help="Adopting mix precision training.")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true", help="occupy GPU memory first for training.")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # det args
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # tracking args, added by F&F attack
    parser.add_argument("--disable_strict_birth", action="store_true", help="disable probation period, spawn a trajectory once it is detected")
    parser.add_argument("--enable_sort_tracker", action="store_true", help="use SORT instead of BYTETrack")

    parser.add_argument("--n_frames_per_seq", type=int, default=30, help="length of each video sequence")
    parser.add_argument("--max_det_num", type=int, default=50000, help="maximal number of detections in one frame")


    # Adversarial args, added by F&F attack
    parser.add_argument('--max_attack_iter', type=int, default=30, help="number of iterations for attacking each frame")
    parser.add_argument('--epsilon', type=float, default=4/255, help="perturbation bound")

    parser.add_argument('--alpha', type=float, default=1/255, help="step length of each iteration")
    parser.add_argument('--adv_grad_clip', type=float, default=0, help="ignore adversarial gradients with tiny amplitudes, e.g., < 1e-5")
    parser.add_argument('--fp_shift_ratio', type=float, default=0.2, help="shift parameter, i.e., $kappa$ in the paper")
    parser.add_argument('--fp_scale_ratio', type=float, default=0.8, help="scale parameter, i.e., $s$ in the paper")
    parser.add_argument('--attack_interval', type=int, default=8, help="conduct attack every [attack_interval] frames, F&F attacks each frame if attack_interval==1, and conduct no attack if attack_interval==0")  # 每多少帧攻击一帧, 等于1则表示对每一帧都攻击, 等于0则表示不攻击
    parser.add_argument('--attack_step_per_interval', type=int, default=1, help="attack how many consecutive frames")
    parser.add_argument('--attack_mode', type=int, default=6)  # option: 6

    # debug, added by F&F attack
    parser.add_argument('--debug', type=int, default=0, help="set debug==6 to turn on visualization")

    # adv_loss_weight, added by F&F attack
    parser.add_argument('--adv_conf_loss_weight', type=float, default=0)
    parser.add_argument('--adv_cls_loss_weight', type=float, default=0)
    parser.add_argument('--adv_iou_loss_weight', type=float, default=0)
    parser.add_argument('--adv_l1_loss_weight', type=float, default=0)

    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        # random.seed(exp.seed)
        # torch.manual_seed(exp.seed)
        # cudnn.deterministic = True
        torch.manual_seed(exp.seed)
        torch.cuda.manual_seed(exp.seed)
        torch.cuda.manual_seed_all(exp.seed)
        np.random.seed(exp.seed)
        random.seed(exp.seed)
        torch.backends.cudnn.deterministic = True

        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True
    cudnn.benchmark = True

    joint_attacker_and_tracker = Joint_Attacker_and_Tracker(exp, args)
    joint_attacker_and_tracker.attack_and_track()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    suffex = '_SORT' if args.enable_sort_tracker else ''
    if args.n_frames_per_seq != 0:
        # exp.val_ann = f'val_half_30_frames_each.json'  # gt
        exp.val_ann = f'val_half_with_private_det_{args.n_frames_per_seq}_frames_per_seq{suffex}.json'  # private det
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )
