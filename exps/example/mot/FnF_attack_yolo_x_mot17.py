# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.val_ann = ""
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.test_conf = 0.1
        self.nmsthre = 0.7

    def get_adv_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, AdvTransform

        advdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "mot17_data"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=AdvTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                advdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(advdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        adv_loader = torch.utils.data.DataLoader(advdataset, **dataloader_kwargs)

        return adv_loader