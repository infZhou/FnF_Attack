# F&F Attack: Adversarial Attack against Multiple Object Trackers by Inducing False Negatives and False Positives

PyTorch Implementation for paper [F&F Attack: Adversarial Attack against Multiple Object Trackers by Inducing
False Negatives and False Positives]((https://infzhou.github.io/FnFAttack/index.html)), ICCV 2023.


#### [Project Page](https://infzhou.github.io/FnFAttack/index.html) | [Paper](https://infzhou.github.io/folder/ZhouTao_F&F_Attack_ICCV2023_main_text+supp.pdf) | [Video](https://infzhou.github.io/folder/Video_F&F_Attack_ICCV_2023.mp4)

## Abstract
Multi-object trackers that follow the tracking-by-detection paradigm heavily rely on detectors. Our work reveals the vulnerability of such trackers to detection attackers. We propose an error detection mechanism that can effectively induce identity switching. By only fooling the detection module and leaving the association module as a black box, we achieve 75%, and 95% attack success rates on attacking CenterTrack (by targeting 1 frame) and ByteTrack (by targeting 3 frames), respectively. The released version contains the attack on BYTETrack. 


## Installation
Step1. Please setup [BYTETrack](https://github.com/ifzhang/ByteTrack) following its instructions. Our project has been tested on PyTorch version `1.8.0`. We observed potential issues with GPU memory leaks using a higher version of PyTorch (e.g., `1.13.0`). We may fix this problem in future updates. Our released project requires at least 8GB GPU memory (depending on the size of the targeted model).

Setp2. Download the code of F&F Attack and merge it with BYTETrack where several files of BYTETrack will be overwritten.

## Data preparation
Download [MOT17](https://motchallenge.net/) and [MOT20](https://motchallenge.net/), and organize them as the following structure:
```
/path/to/your/own/data/dir/
   |——————MOT17
   |        └——————train
   |        └——————test
   └——————MOT20
   |        └——————train
   |        └——————test
```
Then, you need to change the variables `img_dir_17` and `img_dir_20` in `/yolox/data/datasets/mot.py` to your data path. 

As mentioned in our paper, to enrich the sequence, we split each evaluation sequence into segments every 30 frames. Additionally, to simplify the code, the released version uses offline files to load the tracking results of BYTETrack obtained on clean images. For convenience, we provide the [processed data](https://drive.google.com/file/d/1fBnwMUI1myLYvA3ezHVY770CY4Y8Slzo/view?usp=sharing) that has already been converted to COCO format. Please download it and place it as follows:
```
datasets
   |——————mot17_data
   |        └——————annotations
   |        └——————gt_txts
   |        └——————seqinfo
   |        └——————seqmap
   |——————mot20_data
   |        └——————annotations
   |        └——————gt_txts
   |        └——————seqinfo
   |        └——————seqmap
```

## Model zoo
|Model    |  Link |  Description |
|----------------|------|----------|
|MOT17_half|[[google]](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view?usp=sharing)| Trained on CrowdHuman and MOT17 half train (provided by BYTETrack)|
|MOT20_half|[[google]](https://drive.google.com/file/d/1uVXiDJDBbQ-EoyYQVfBGyZMuAqnQkjzh/view?usp=drive_link)| Trained on CrowdHuman and MOT20 half train|

## Attacking
* **Attack BYTETrack on MOT17 validation set**
```shell
cd <FnF_Attack_HOME>
python3 tools/joint_attack_and_track.py -expn attack_mot17_BYTETrack -f exps/example/mot/FnF_attack_yolo_x_mot17.py -c models/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --attack_mode 6 --adv_conf_loss_weight 1 --adv_cls_loss_weight 0 --adv_iou_loss_weight 0 --adv_l1_loss_weight 1 --attack_interval 8 --attack_step_per_interval 3 --n_frames_per_seq 30 --adv_grad_clip 1e-5 --max_attack_iter 30 --fp_shift_ratio 0.3 --fp_scale_ratio 0.4
```
* **Attack SORT on MOT17 validation set**
```shell
cd <FnF_Attack_HOME>
python3 tools/joint_attack_and_track.py -expn attack_mot17_SORT -f exps/example/mot/FnF_attack_yolo_x_mot17.py -c models/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --attack_mode 6 --adv_conf_loss_weight 1 --adv_cls_loss_weight 0 --adv_iou_loss_weight 0 --adv_l1_loss_weight 1 --attack_interval 8 --attack_step_per_interval 3 --n_frames_per_seq 30 --adv_grad_clip 1e-5 --max_attack_iter 30 --fp_shift_ratio 0.2 --fp_scale_ratio 0.6 --enable_sort_tracker
```
* **Attack BYTETrack on MOT20 validation set**
```shell
cd <FnF_Attack_HOME>
python3 tools/joint_attack_and_track.py -expn attack_mot20_BYTETrack -f exps/example/mot/FnF_attack_yolo_x_mot20.py -c models/mot20_ablation.pth.tar --mot20 -b 1 -d 1 --fp16 --fuse --attack_mode 6 --adv_conf_loss_weight 1 --adv_cls_loss_weight 0 --adv_iou_loss_weight 0 --adv_l1_loss_weight 1 --attack_interval 8 --attack_step_per_interval 3 --n_frames_per_seq 30 --adv_grad_clip 1e-5 --max_attack_iter 30 --fp_shift_ratio 0.3 --fp_scale_ratio 0.4
```
* **Attack SORT on MOT20 validation set**
```shell
cd <FnF_Attack_HOME>
python3 tools/joint_attack_and_track.py -expn attack_mot20_SORT -f exps/example/mot/FnF_attack_yolo_x_mot20.py -c models/mot20_ablation.pth.tar --mot20 -b 1 -d 1 --fp16 --fuse --attack_mode 6 --adv_conf_loss_weight 1 --adv_cls_loss_weight 0 --adv_iou_loss_weight 0 --adv_l1_loss_weight 1 --attack_interval 8 --attack_step_per_interval 3 --n_frames_per_seq 30 --adv_grad_clip 1e-5 --max_attack_iter 30 --fp_shift_ratio 0.2 --fp_scale_ratio 0.6 --enable_sort_tracker
```
## Acknowledgement

A large part of the code is borrowed from [BYTETrack](https://github.com/ifzhang/ByteTrack). Many thanks for their wonderful works.

## Citation
```
@inproceedings{zhou2023fnf_attack,
  title={F\&F Attack: Adversarial Attack against Multiple Object Trackers by Inducing False Negatives and False Positives},
  author={Zhou, Tao and Ye, Qi and Luo, Wenhan and Zhang, Kaihao and Shi, Zhiguo and Chen, Jiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4573--4583},
  year={2023}
}
```



