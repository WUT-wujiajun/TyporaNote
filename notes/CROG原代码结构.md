CROG-Code

# Train_crog.py

```
import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial
from collections import OrderedDict

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = '99ee90fdefff711f21b8b40a0fac1bdb95da2aa5'


import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.dataset import OCIDVLGDataset
from engine.crog_engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_crog
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    torch.multiprocessing.set_start_method('spawn')
    
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    # mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ), join=True)
    
    children = []
    for i in range(args.world_size):
        subproc = mp.Process(target=main_worker, args=(i, args))
        children.append(subproc)
        subproc.start()

    for i in range(args.world_size):
        children[i].join()


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    # wandb
    # if args.rank == 0:
    #     wandb.init(job_type="training",
    #                mode="online",
    #                config=args,
    #                project="CROG",
    #                name=args.exp_name,
    #                tags=[args.dataset, args.clip_pretrain])
    dist.barrier()

    # build model
    model, param_list = build_crog(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    logger.info(args)
    
    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()
    
    # # resume
    # best_IoU = 0.0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(
    #             args.resume, map_location=torch.device('cpu'))
    #         args.start_epoch = checkpoint['epoch']
    #         best_IoU = checkpoint["best_iou"]
    #         state_dict = checkpoint['state_dict']
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = k[7:] # remove `module.`
    #             new_state_dict[name] = v
    #         # load params
    #         model.load_state_dict(new_state_dict)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.resume, checkpoint['epoch']))
    #     else:
    #         raise ValueError(
    #             "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
    #             .format(args.resume))
    
    
    
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)

    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int(
        (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)

        
    train_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='train',
                            version=args.version)
    val_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='val',
                            version=args.version)
        

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True,
                                   collate_fn=OCIDVLGDataset.collate_fn)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False,
                                 collate_fn=OCIDVLGDataset.collate_fn)

    best_IoU = 0.0
    best_j_index = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(
                args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            best_j_index = checkpoint["best_j_index"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)
        # evaluation
        if args.use_grasp_masks:
            iou, prec_dict, j_index = validate_with_grasp(val_loader, model, epoch_log, args)
        else:
            iou, prec_dict, j_index = validate_without_grasp(val_loader, model, epoch_log, args)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'best_j_index': best_j_index,
                    'prec': prec_dict,
                    'j_index': j_index,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(args.output_dir, "best_iou_model.pth")
                shutil.copyfile(lastname, bestname)
            
            if j_index[0] >= best_j_index:
                best_j_index = j_index[0]
                bestname = os.path.join(args.output_dir, "best_jindex_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)
    # if dist.get_rank() == 0:
    #     wandb.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
```

# 1.models

## 1.clip.py

```
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2d(stride)),
                             ("0",
                              nn.Conv2d(inplanes,
                                        planes * self.expansion,
                                        1,
                                        stride=1,
                                        bias=False)),
                             ("1", nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.spacial_dim = spacial_dim
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        # residual
        self.connect = nn.Sequential(
            nn.Conv2d(embed_dim, output_dim, 1, stride=1, bias=False),
            nn.BatchNorm2d(output_dim))

    def resize_pos_embed(self, pos_embed, input_shpae):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, C, L_new]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h = pos_w = self.spacial_dim
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight,
                                         size=input_shpae,
                                         align_corners=False,
                                         mode='bicubic')
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        # pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed_weight.transpose(-2, -1)

    def forward(self, x):
        B, C, H, W = x.size()
        res = self.connect(x)
        x = x.reshape(B, C, -1)  # NC(HW)
        # x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(1+HW)
        pos_embed = self.positional_embedding.unsqueeze(0)
        pos_embed = self.resize_pos_embed(pos_embed, (H, W))  # NC(HW)
        x = x + pos_embed.to(x.dtype)  # NC(HW)
        x = x.permute(2, 0, 1)  # (HW)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        x = x.permute(1, 2, 0).reshape(B, -1, H, W)
        x = x + res
        x = F.relu(x, True)

        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x4 = self.attnpool(x4)

        return (x2, x3, x4)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            # vision
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,
            # text
            context_length: int,
            txt_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers,
                                         output_dim=embed_dim,
                                         heads=vision_heads,
                                         input_resolution=image_resolution,
                                         width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads,
                                            output_dim=embed_dim)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(txt_length))

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.token_embedding.requires_grad_ = False
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width**-0.5)

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)[:x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = x[torch.arange(x.shape[0]),
                  text.argmax(dim=-1)] @ self.text_projection
        # x = x @ self.text_projection
        # state = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return x, state

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                    *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                    "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, txt_length: int, load_weights=True):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2] for k in state_dict
                    if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] -
             1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            "visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith(f"transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, txt_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if load_weights:
        convert_weights(model)
        model.load_state_dict(state_dict, False)
        
    return model.eval()

```

## 2.crog.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder, MultiTaskProjector


class CROG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Flags for ablation study
        self.use_contrastive = cfg.use_contrastive
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_grasp_masks = cfg.use_grasp_masks
        
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, self.use_pretrained_clip).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        # Decoder
        if self.use_contrastive:
            print("Use contrastive learning module")
            # Decoder
            self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                            d_model=cfg.vis_dim,
                                            nhead=cfg.num_head,
                                            dim_ffn=cfg.dim_ffn,
                                            dropout=cfg.dropout,
                                            return_intermediate=cfg.intermediate)
        else:
            print("Disable contrastive learning module")
        if self.use_grasp_masks:
            # Projector
            print("Use grasp masks")
            self.proj = MultiTaskProjector(cfg.word_dim, cfg.vis_dim // 2, 3)
        else:
            print("Disable grasp masks")
            self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        
        if self.use_contrastive:
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)

        if self.use_grasp_masks:
            
            # b, 1, 104, 104
            pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                    grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
                    grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
                    grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
                    grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()

                # Ratio Augmentation
                total_area = mask.shape[2] * mask.shape[3]
                coef = 1 - (mask.sum(dim=(2,3)) / total_area)

                # Generate weight
                weight = mask * 0.5 + 1

                loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
                grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
                grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
                grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

                # @TODO adjust coef of different loss items
                total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss

                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = grasp_qua_loss.item()
                loss_dict["m_sin"] = grasp_sin_loss.item()
                loss_dict["m_cos"] = grasp_cos_loss.item()
                loss_dict["m_wid"] = grasp_wid_loss.item()

                # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
                # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
            else:
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        else:
            # b, 1, 104, 104
            pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                        mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = 0
                loss_dict["m_sin"] = 0
                loss_dict["m_cos"] = 0
                loss_dict["m_wid"] = 0
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), loss, loss_dict
            else:
                return pred.detach(), mask
```

## 3.layer.py

```
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class MultiTaskProjector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim*5, 1))

        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        x = torch.tensor_split(x, 5, dim=1) # no tensor_split api in torch 1.7, please use it in higher version
        # x = torch.chunk(x, 5, dim=1)

        mask_x = x[0]
        grasp_qua_x = x[1]
        grasp_sin_x = x[2]
        grasp_cos_x = x[3]
        grasp_wid_x = x[4]

        B, C, H, W = mask_x.size()


        # 1, b*256, 104, 104
        mask_x = mask_x.reshape(1, B * C, H, W)
        grasp_qua_x = grasp_qua_x.reshape(1, B * C, H, W)
        grasp_sin_x = grasp_sin_x.reshape(1, B * C, H, W)
        grasp_cos_x = grasp_cos_x.reshape(1, B * C, H, W)
        grasp_wid_x = grasp_wid_x.reshape(1, B * C, H, W)


        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        mask_out = F.conv2d(mask_x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        
        grasp_qua_out = F.conv2d(grasp_qua_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
        grasp_sin_out = F.conv2d(grasp_sin_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)

        grasp_cos_out = F.conv2d(grasp_cos_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
        grasp_wid_out = F.conv2d(grasp_wid_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
            
        mask_out = mask_out.transpose(0, 1)
        grasp_qua_out = grasp_qua_out.transpose(0, 1)
        grasp_sin_out = grasp_sin_out.transpose(0, 1)
        grasp_cos_out = grasp_cos_out.transpose(0, 1)
        grasp_wid_out = grasp_wid_out.transpose(0, 1)
        # b, 1, 104, 104

        return mask_out, grasp_qua_out, grasp_sin_out, grasp_cos_out, grasp_wid_out


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
            -1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq

```

## 4.ssg.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2

from collections import OrderedDict
from math import sqrt
import numpy as np
from itertools import product
from utils.box_utils import match, crop, ones_crop, make_anchors



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       self.norm_layer(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample, self.norm_layer)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        print(f'\nBackbone is initiated with {path}.\n')


class PredictionModule(nn.Module):
    def __init__(self, cfg, coef_dim=32):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim
        self.gr_coef_dim = coef_dim / 2

        self.upfeature = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.bbox_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        self.coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        if cfg.with_grasp_masks:
            # Generate 4 grasp masks
            self.grasp_coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim * 4,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())


    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)

        grasp_coef_layer = self.grasp_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4, self.coef_dim)

        
        return conf, box, coef, grasp_coef_layer


class ProtoNet(nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.proto1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proto2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True)),
                                                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True))])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p5 = self.pred_layers[2](p5_1)
        p4 = self.pred_layers[1](p4_1)
        p3 = self.pred_layers[0](p3_1)

        p6 = self.downsample_layers[0](p5)
        p7 = self.downsample_layers[1](p6)

        return p3, p4, p5, p6, p7


class SSG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        if cfg.backbone == "resnet":
            self.backbone = ResNet(layers=cfg.resnet_layers)
            if cfg.path_to_pretrained_resnet and not cfg.resume:
                self.backbone.init_backbone(cfg.path_to_pretrained_resnet)
            if cfg.with_depth:
                with torch.no_grad():
                    # Add extra depth channel for net.backbone
                    weight = self.backbone.conv1.weight.clone()
                    self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    self.backbone.conv1.weight[:,:3] = weight
            self.fpn = FPN(in_channels=cfg.fpn_in_channels)

        else:
            raise NotImplementedError
        
        self.proto_net = ProtoNet(coef_dim=cfg.num_protos)
        self.prediction_layers = PredictionModule(cfg, coef_dim=cfg.num_protos)

        self.anchors = []
        scales = [int(cfg.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in cfg.anchor_strides]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(cfg, size, size, scales[i])

        if self.training:
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes, kernel_size=1)

        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    
    def forward(self, data_dict):
        if self.cfg.with_depth:
            img = torch.cat([data_dict["rgb"], data_dict["depth"]], dim=1)
        else:
            img = data_dict["rgb"]
        x = self.backbone(img)
        x = self.fpn(x[1:4])
        protos = self.proto_net(x[0])
        protos = protos.permute(0, 2, 3, 1).contiguous()

        class_pred, box_pred, ins_coef_pred, grasp_coef_pred = [], [], [], []

        for aa in x:
            class_p, box_p, ing_coef_p, grasp_coef_p = self.prediction_layers(aa)
            class_pred.append(class_p)
            box_pred.append(box_p)
            ins_coef_pred.append(ing_coef_p)
            grasp_coef_pred.append(grasp_coef_p)
        
        class_pred = torch.cat(class_pred, dim=1)
        box_pred = torch.cat(box_pred, dim=1)
        ins_coef_pred = torch.cat(ins_coef_pred, dim=1)
        grasp_coef_pred = torch.cat(grasp_coef_pred, dim=1)
        
        output_dict = {
                "anchors": self.anchors,
                "protos": protos,
                "cls_pred": F.softmax(class_pred, -1),
                "box_pred": box_pred,
                "ins_coef_pred": ins_coef_pred,
                "grasp_coef_pred": grasp_coef_pred,
            }

        if self.training:
            seg_pred = self.semantic_seg_conv(x[0])
            loss_dict = self.compute_loss(
                class_pred, 
                box_pred, 
                ins_coef_pred, 
                grasp_coef_pred, 
                protos, seg_pred, 
                data_dict, output_dict
            )
            return output_dict, loss_dict
        else:
            return output_dict
    


    def compute_loss(
        self, 
        class_pred, box_pred, ins_coef_pred, grasp_coef_pred, 
        protos, seg_pred, 
        data_dict, output_dict
    ):
        device = class_pred.device
        class_gt = [None] * len(data_dict["bboxes"])
        batch_size = box_pred.size(0)

        if isinstance(self.anchors, list):
            self.anchors = torch.tensor(self.anchors, device=device).reshape(-1, 4)

        num_anchors = self.anchors.shape[0]

        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        for i in range(batch_size):
            box_gt = data_dict["bboxes"][i][:, :-1]
            class_gt[i] = data_dict["bboxes"][i][:, -1].long()
            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                  self.anchors, class_gt[i])

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and \
               (not anchor_max_i.requires_grad), 'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        pos_bool = conf_gt > 0

        loss_c = self.category_loss(class_pred, conf_gt, pos_bool)
        loss_b = self.box_loss(box_pred, all_offsets, pos_bool)
        if self.cfg.intermidiate_output:
            loss_m = self.lincomb_mask_loss(ins_coef_pred, protos, data_dict["ins_masks"], pos_bool, anchor_max_i, anchor_max_gt, output_dict)
        else:
            loss_m = self.lincomb_mask_loss(ins_coef_pred, protos, data_dict["ins_masks"], pos_bool, anchor_max_i, anchor_max_gt)
        loss_g = self.lincomb_grasp_masks_loss(grasp_coef_pred, protos, data_dict["grasp_masks"], pos_bool, anchor_max_i, anchor_max_gt)
        loss_s = self.semantic_seg_loss(seg_pred, data_dict["sem_mask"], data_dict["labels"])

        return {
            "loss_cls": loss_c,
            "loss_box": loss_b,
            "loss_ins": loss_m,
            "loss_sem": loss_s,
            "loss_qua": loss_g["qua"],
            "loss_sin": loss_g["sin"],
            "loss_cos": loss_g["cos"],
            "loss_wid": loss_g["wid"]
        }

    def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)

        batch_conf_max = batch_conf.max()

        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + neg_bool)]

        return self.cfg.alpha_conf * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

    
    def box_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]

        return self.cfg.alpha_bbox * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos

    

    def lincomb_mask_loss(self, ins_coef_p, protos, ins_masks_gt, pos_bool, anchor_max_i, anchor_max_gt, output_dict=None):
        proto_h, proto_w = protos.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0

        inter_mask_p = []
        inter_mask_gt = []

        for i in range(ins_coef_p.shape[0]):
            downsampled_masks = F.interpolate(ins_masks_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = ins_coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:
                continue
            
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]
            
            num_pos = pos_coef.size(0)
            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            mask_p = torch.sigmoid(protos[i] @ pos_coef.t())
            mask_p = crop(mask_p, pos_anchor_box)

            if output_dict is not None and self.cfg.intermidiate_output:
                inter_mask_p.append(mask_p.data)
                inter_mask_gt.append(pos_mask_gt.data)

            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')

            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos
            
            loss_m += torch.sum(mask_loss)
        
        if len(inter_mask_p) > 0 and output_dict is not None:
            inter_mask_p = torch.cat(inter_mask_p, dim=-1).permute(2, 0, 1)
            inter_mask_gt = torch.cat(inter_mask_gt, dim=-1).permute(2, 0, 1)

            output_dict["inter_mask_p"] = inter_mask_p
            output_dict["inter_mask_gt"] = inter_mask_gt


        return self.cfg.alpha_ins * loss_m / proto_h / proto_w / total_pos_num

    

    def lincomb_grasp_masks_loss(self, grasp_coef_p, protos, grasp_masks_gt, pos_bool, anchor_max_i, anchor_max_gt):
        proto_h, proto_w = protos.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_dict = {
            "qua": 0.0,
            "sin": 0.0,
            "cos": 0.0,
            "wid": 0.0
        }
        for i in range(grasp_coef_p.shape[0]):
            for idx, key in enumerate(grasp_masks_gt.keys()):
                downsampled_masks = F.interpolate(grasp_masks_gt[key][i].unsqueeze(0), (proto_h, proto_w), mode='bilinear', align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                pos_anchor_i = anchor_max_i[i][pos_bool[i]]
                pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
                pos_coef = grasp_coef_p[i, pos_bool[i], idx, :]

                if pos_anchor_i.size(0) == 0:
                    continue
                
                old_num_pos = pos_coef.size(0)
                if old_num_pos > self.cfg.masks_to_train:
                    perm = torch.randperm(pos_coef.size(0))
                    select = perm[:self.cfg.masks_to_train]
                    pos_coef = pos_coef[select]
                    pos_anchor_i = pos_anchor_i[select]
                    pos_anchor_box = pos_anchor_box[select]
                
                num_pos = pos_coef.size(0)

                pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

                mask_p = torch.sigmoid(protos[i] @ pos_coef.t())
                if key == "cos":
                    mask_p = ones_crop(mask_p, pos_anchor_box)
                else:
                    mask_p = crop(mask_p, pos_anchor_box) 
                    # if key == "qua":
                    #     for j in range(mask_p.shape[-1]):
                    #         m_p = (mask_p[:, :, j].data.cpu().numpy() * 255).astype(int)
                    #         m_g = (pos_mask_gt[:, :, j].data.cpu().numpy() * 255).astype(int)
                    #         cv2.imwrite(f"./test/{j}_qua_p.png", m_p)
                    #         cv2.imwrite(f"./test/{j}_qua_g.png", m_g)
                loss = F.smooth_l1_loss(mask_p, pos_mask_gt, reduction="none")
                anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
                loss = loss.sum(dim=(0, 1)) / anchor_area
                
                if old_num_pos > num_pos:
                    loss *= old_num_pos / num_pos
                
                loss_dict[key] += self.cfg.alpha_grasp * torch.sum(loss) / proto_h /proto_w / total_pos_num
        
        return loss_dict
    

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0).unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return self.cfg.alpha_sem * loss_s / mask_h / mask_w / batch_size
```

# 2.tools

## 1.data_process.py

```
import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from refer import REFER

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset',
                    type=str,
                    choices=['refcoco', 'refcoco+', 'refcocog', 'refclef'],
                    default='refcoco')
parser.add_argument('--split', type=str, default='umd')
parser.add_argument('--generate_mask', action='store_true')
args = parser.parse_args()
img_path = os.path.join(args.data_root, 'images', 'train2014')

h, w = (416, 416)

refer = REFER(args.data_root, args.dataset, args.split)

print('dataset [%s_%s] contains: ' % (args.dataset, args.split))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print('%s expressions for %s refs in %s images.' %
      (len(refer.Sents), len(ref_ids), len(image_ids)))

print('\nAmong them:')
if args.dataset == 'refclef':
    if args.split == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif args.dataset == 'refcoco':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcoco+':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcocog':
    splits = ['train', 'val',
              'test']  # we don't have test split for refcocog right now.

for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print('%s refs are in split [%s].' % (len(ref_ids), split))


def cat_process(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
    ann_path = os.path.join(output_dir, 'anns', dataset)
    mask_path = os.path.join(output_dir, 'masks', dataset)
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for split in splits:
        dataset_array = []
        ref_ids = refer.getRefIds(split=split)
        print('Processing split:{} - Len: {}'.format(split, len(ref_ids)))
        for i in tqdm(ref_ids):
            ref_dict = {}

            refs = refer.Refs[i]
            bboxs = refer.getRefBox(i)
            sentences = refs['sentences']
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            cat = cat_process(refs['category_id'])
            image_urls = image_urls['file_name']
            if dataset == 'refclef' and image_urls in [
                    '19579.jpg', '17975.jpg', '19575.jpg'
            ]:
                continue
            box_info = bbox_process(bboxs)

            ref_dict['bbox'] = box_info
            ref_dict['cat'] = cat
            ref_dict['segment_id'] = i
            ref_dict['img_name'] = image_urls

            if generate_mask:
                cv2.imwrite(os.path.join(mask_path,
                                         str(i) + '.png'),
                            refer.getMask(refs)['mask'] * 255)

            sent_dict = []
            for i, sent in enumerate(sentences):
                sent_dict.append({
                    'idx': i,
                    'sent_id': sent['sent_id'],
                    'sent': sent['sent'].strip()
                })

            ref_dict['sentences'] = sent_dict
            ref_dict['sentences_num'] = len(sent_dict)

            dataset_array.append(ref_dict)
        print('Dumping json file...')
        with open(os.path.join(output_dir, 'anns', dataset, split + '.json'),
                  'w') as f:
            json.dump(dataset_array, f)


prepare_dataset(args.dataset, splits, args.output_dir, args.generate_mask)

```

## 2.folder2lmdb.py

```
import argparse
import os
import os.path as osp
import lmdb
import pyarrow as pa
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(json_data, img_dir, mask_dir, output_dir, split, write_frequency=1000):
    lmdb_path = osp.join(output_dir, "%s.lmdb" % split)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)

    txn = db.begin(write=True)
    tbar = tqdm(json_data)
    for idx, item in enumerate(tbar):
        img = raw_reader(osp.join(img_dir, item['img_name']))
        mask = raw_reader(osp.join(mask_dir, f"{item['segment_id']}.png"))
        data = {
            'img': img, 
            'mask': mask, 
            'cat': item['cat'],
            'seg_id': item['segment_id'], 
            'img_name': item['img_name'],
            'num_sents': len(item['sentences']), 
            'sents': [i['sent'] for i in item['sentences']]
        }
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(data))
        if idx % write_frequency == 0:
            # print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def parse_args():
    parser = argparse.ArgumentParser(description='COCO Folder to LMDB.')
    parser.add_argument('-j', '--json-dir', type=str,
                        default='',
                        help='the name of json file.')
    parser.add_argument('-i', '--img-dir', type=str,
                        default='refcoco+',
                        help='the folder of images.')
    parser.add_argument('-m', '--mask-dir', type=str,
                        default='refcoco+',
                        help='the folder of masks.')
    parser.add_argument('-o', '--output-dir', type=str,
                        default='refcoco+',
                        help='the folder of output lmdb file.')
    parser.add_argument('-s', '--split', type=str,
                        default='train',
                        help='the split type.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.split = osp.basename(args.json_dir).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_dir, 'rb') as f:
        json_data = json.load(f)

    folder2lmdb(json_data, args.img_dir, args.mask_dir, args.output_dir, args.split)

```

## 3.latency.py

```
import argparse
import sys
import time
import warnings

sys.path.append('./')
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
import utils.config as config
from model import build_segmenter


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # init arguments
    args = get_parser()
    torch.cuda.set_device(0)
    # create model
    model, _ = build_segmenter(args)
    model = model.cuda()
    model.eval()
    # set cudnn state
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    # init dummy tensor
    image = torch.randn(1, 3, 416, 416).cuda()
    text = torch.randint(4096, size=(1, args.word_len)).long().cuda()
    # init time & memory
    avg_time = 0
    avg_mem = 0
    # record initial gpu memory
    mem = torch.cuda.max_memory_allocated()

    with torch.no_grad():
        for i in range(500):
            start_time = time.time()
            _ = model(image, text)
            torch.cuda.synchronize()
            if (i+1) >= 100:
                avg_time += (time.time() - start_time)
                avg_mem += (torch.cuda.max_memory_allocated() - mem) / 1.073742e9
    params = count_parameters(model) * 1e-6
    print('#########################################')
    print("Average Parameters : {:.2f} M".format(params))
    print("Average FPS: {:.2f}".format(400/avg_time))
    print("Average GPU Memory: {:.2f} GB".format(avg_mem/400))
    print('#########################################')


if __name__ == '__main__':
    main()

```

## 4.prepare_datasets.md

````
## Prepare datasets

In our paper, we conduct experiments on three common-used datasets, including Ref-COCO, Ref-COCO+ and G-Ref.

### 1. COCO 2014

The data could be found at [here](https://cocodataset.org/#download). Please run the following commands to download.

```shell
# download
mkdir datasets && cd datasets
wget http://images.cocodataset.org/zips/train2014.zip

# unzip
unzip train2014.zip -d images/ && rm train2014.zip

```

### 2. Ref-COCO

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

# unzip
unzip refcoco.zip && rm refcoco.zip

# convert
python ../tools/data_process.py --data_root . --output_dir . --dataset refcoco --split unc --generate_mask

# lmdb
python ../tools/folder2lmdb.py -j anns/refcoco/train.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/val.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/testA.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../tools/folder2lmdb.py -j anns/refcoco/testB.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco

# clean
rm -r refcoco

```

### 3. Ref-COCO+

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip

# unzip
unzip refcoco+.zip && rm refcoco+.zip

# convert
python ../tools/data_process.py --data_root . --output_dir . --dataset refcoco+ --split unc --generate_mask

# lmdb
python ../tools/folder2lmdb.py -j anns/refcoco+/train.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+
python ../tools/folder2lmdb.py -j anns/refcoco+/val.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+
python ../tools/folder2lmdb.py -j anns/refcoco+/testA.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+
python ../tools/folder2lmdb.py -j anns/refcoco+/testB.json -i images/train2014/ -m masks/refcoco+ -o lmdb/refcoco+

# clean
rm -r refcoco+

```

### 4. Ref-COCOg

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.
(Note that we adopt two different splits of this dataset, 'umd' and 'google'.)

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

# unzip
unzip refcocog.zip && rm refcocog.zip

# convert
python ../tools/data_process.py --data_root . --output_dir . --dataset refcocog --split umd --generate_mask  # umd split
mv anns/refcocog anns/refcocog_u
mv masks/refcocog masks/refcocog_u

python ../tools/data_process.py --data_root . --output_dir . --dataset refcocog --split google --generate_mask  # google split
mv anns/refcocog anns/refcocog_g
mv masks/refcocog masks/refcocog_g

# lmdb
python ../tools/folder2lmdb.py -j anns/refcocog_u/train.json -i images/train2014/ -m masks/refcocog_u -o lmdb/refcocog_u
python ../tools/folder2lmdb.py -j anns/refcocog_u/val.json -i images/train2014/ -m masks/refcocog_u -o lmdb/refcocog_u
python ../tools/folder2lmdb.py -j anns/refcocog_u/test.json -i images/train2014/ -m masks/refcocog_u -o lmdb/refcocog_u

python ../tools/folder2lmdb.py -j anns/refcocog_g/train.json -i images/train2014/ -m masks/refcocog_g -o lmdb/refcocog_g
python ../tools/folder2lmdb.py -j anns/refcocog_g/val.json -i images/train2014/ -m masks/refcocog_g -o lmdb/refcocog_g

rm -r refcocog

```

### 5. Datasets struture

After the above-mentioned commands, the strutre of the dataset folder should be like:

```none
datasets
├── anns
│   ├── refcoco
│   │   ├── xxx.json
│   ├── refcoco+
│   │   ├── xxx.json
│   ├── refcocog_g
│   │   ├── xxx.json
│   ├── refcocog_u
│   │   ├── xxx.json
├── images
│   ├── train2014
│   │   ├── xxx.jpg
├── lmdb
│   ├── refcoco
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
│   ├── refcoco+
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
│   ├── refcocog_g
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
│   ├── refcocog_u
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
├── masks
│   ├── refcoco
│   │   ├── xxx.png
│   ├── refcoco+
│   │   ├── xxx.png
│   ├── refcocog_g
│   │   ├── xxx.png
│   ├── refcocog_u
│   │   ├── xxx.png

```
````

## 5.refer.py

```
__author__ = 'licheng'
"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google
The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import itertools
import json
import os.path as osp
import pickle
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pycocotools import mask


class REFER:
    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset

        self.data['refs'] = pickle.load(open(ref_file, 'rb'), fix_imports=True)

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'],
                                                       []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']
                            ]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs
                            if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [
                    self.imgToAnns[image_id] for image_id in image_ids
                    if image_id in self.imgToAnns
                ]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(
                    set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(
                set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons,
                                    facecolors=color,
                                    edgecolors=(1, 1, 0, 0),
                                    linewidths=3,
                                    alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons,
                                    facecolors=color,
                                    edgecolors=(1, 0, 0, 0),
                                    linewidths=1,
                                    alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]),
                                 bbox[2],
                                 bbox[3],
                                 fill=False,
                                 edgecolor='green',
                                 linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'],
                                   image['width'])
        else:
            rle = ann['segmentation']

        # for i in range(len(rle['counts'])):
        # print(rle)
        m = mask.decode(rle)
        m = np.sum(
            m, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}
        # # position
        # position_x = np.mean(np.where(m==1)[1]) # [1] means columns (matlab style) -> x (c style)
        # position_y = np.mean(np.where(m==1)[0]) # [0] means rows (matlab style)    -> y (c style)
        # # mass position (if there were multiple regions, we use the largest one.)
        # label_m = label(m, connectivity=m.ndim)
        # regions = regionprops(label_m)
        # if len(regions) > 0:
        # 	largest_id = np.argmax(np.array([props.filled_area for props in regions]))
        # 	largest_props = regions[largest_id]
        # 	mass_y, mass_x = largest_props.centroid
        # else:
        # 	mass_x, mass_y = position_x, position_y
        # # if centroid is not in mask, we find the closest point to it from mask
        # if m[mass_y, mass_x] != 1:
        # 	print 'Finding closes mask point ...'
        # 	kernel = np.ones((10, 10),np.uint8)
        # 	me = cv2.erode(m, kernel, iterations = 1)
        # 	points = zip(np.where(me == 1)[0].tolist(), np.where(me == 1)[1].tolist())  # row, col style
        # 	points = np.array(points)
        # 	dist   = np.sum((points - (mass_y, mass_x))**2, axis=1)
        # 	id     = np.argsort(dist)[0]
        # 	mass_y, mass_x = points[id]
        # 	# return
        # return {'mask': m, 'area': area, 'position_x': position_x, 'position_y': position_y, 'mass_x': mass_x, 'mass_y': mass_y}
        # # show image and mask
        # I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        # plt.figure()
        # plt.imshow(I)
        # ax = plt.gca()
        # img = np.ones( (m.shape[0], m.shape[1], 3) )
        # color_mask = np.array([2.0,166.0,101.0])/255
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # ax.imshow(np.dstack( (img, m*0.5) ))
        # plt.show()

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


if __name__ == '__main__':
    refer = REFER(dataset='refcocog', splitBy='google')
    ref_ids = refer.getRefIds()
    print(len(ref_ids))

    print(len(refer.Imgs))
    print(len(refer.imgToRefs))

    ref_ids = refer.getRefIds(split='train')
    print('There are %s training referred objects.' % len(ref_ids))

    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if len(ref['sentences']) < 2:
            continue

        pprint(ref)
        print('The label is %s.' % refer.Cats[ref['category_id']])
        plt.figure()
        refer.showRef(ref, seg_box='box')
        plt.show()

        # plt.figure()
        # refer.showMask(ref)
        # plt.show()

```

# 3.utils

## 1.augmentation.py

```
import cv2
from cv2 import resize
import numpy as np
import random


class DataAugmentor(object):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mean = np.asarray([0.48145466, 0.4578275,
                                  0.40821073], dtype=np.float32)
        self.std = np.asarray([0.26862954, 0.26130258,
                                 0.27577711], dtype=np.float32)
        self.mode = mode
    

    def _random_mirror(self, data_dict):
        img = data_dict["rgb"]
        bboxes = data_dict["bboxes"][:, :4]
        if random.randint(0, 1):
            _, width, _ = img.shape
            data_dict["rgb"] = img[:, ::-1]
            data_dict["depth"] = data_dict["depth"][:, ::-1]
            data_dict["ins_masks"] = data_dict["ins_masks"][:, :, ::-1]
            data_dict["grasp_masks"]["qua"] = data_dict["grasp_masks"]["qua"][:, :, ::-1]
            data_dict["grasp_masks"]["ang"] = data_dict["grasp_masks"]["ang"][:, :, ::-1]
            data_dict["grasp_masks"]["wid"] = data_dict["grasp_masks"]["wid"][:, :, ::-1]

            bboxes[:, 0::2] = width - bboxes[:, 2::-2]
            data_dict["bboxes"][:, :4] = bboxes
    

    def _random_brightness(self, img, delta=32):
        img += random.uniform(-delta, delta)
        return np.clip(img, 0., 255.)

    
    def _random_contrast(self, img, lower=0.7, upper=1.3):
        img *= random.uniform(lower, upper)
        return np.clip(img, 0., 255.)

    
    def _random_saturation(self, img, delta=15.):
        img[:, :, 0] += random.uniform(-delta, delta)
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img
    

    def _random_hue(self, img, delta=15.):
        img[:, :, 0] += random.uniform(-delta, delta)
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img

    
    def _photometric_distort(self, data_dict):
        img = data_dict["rgb"].astype(np.float32)
        
        if random.randint(0, 1):
            img = self._random_brightness(img)
        if random.randint(0, 1):
            img = self._random_contrast(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = self._random_saturation(img)
        img = self._random_hue(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0., 255.)

        data_dict["rgb"] = img
    

    def _pad_to_square(self, data_dict):
        img = data_dict["rgb"]
        img_h, img_w = img.shape[:2]
        if img_h == img_w:
            pass
        else:
            pad_size = max(img_h, img_w)
            pad_img = np.zeros((pad_size, pad_size, 3), dtype='float32')
            pad_img[:, :, :] = self.mean
            pad_depth = np.zeros((pad_size, pad_size), dtype='float32')
            pad_ins_masks = np.zeros((data_dict["ins_masks"].shape[0], pad_size, pad_size), dtype='float32')
            pad_qua_masks = np.zeros((data_dict["grasp_masks"]["qua"].shape[0], pad_size, pad_size), dtype='float32')
            pad_ang_masks = np.zeros((data_dict["grasp_masks"]["ang"].shape[0], pad_size, pad_size), dtype='float32')
            pad_wid_masks = np.zeros((data_dict["grasp_masks"]["wid"].shape[0], pad_size, pad_size), dtype='float32')


            if self.mode=="train":
                if img_h < img_w:
                    random_y1 = random.randint(0, img_w - img_h)
                    pad_img[random_y1: random_y1 + img_h, :, :] = data_dict["rgb"]
                    pad_depth[random_y1: random_y1 + img_h, :] = data_dict["depth"]
                    pad_ins_masks[:, random_y1: random_y1 + img_h, :] = data_dict["ins_masks"]
                    pad_qua_masks[:, random_y1: random_y1 + img_h, :] = data_dict["grasp_masks"]["qua"]
                    pad_ang_masks[:, random_y1: random_y1 + img_h, :] = data_dict["grasp_masks"]["ang"]
                    pad_wid_masks[:, random_y1: random_y1 + img_h, :] = data_dict["grasp_masks"]["wid"]
                    data_dict["bboxes"][:, [1, 3]] += random_y1

                if img_h > img_w:
                    random_x1 = random.randint(0, img_h - img_w)
                    pad_img[:, random_x1: random_x1 + img_w, :] = data_dict["rgb"]
                    pad_depth[:, random_x1: random_x1 + img_w] = data_dict["depth"]
                    pad_ins_masks[:, :, random_x1: random_x1 + img_w] = data_dict["ins_masks"]
                    pad_qua_masks[:, :, random_x1: random_x1 + img_w] = data_dict["grasp_masks"]["qua"]
                    pad_ang_masks[:, :, random_x1: random_x1 + img_w] = data_dict["grasp_masks"]["ang"]
                    pad_wid_masks[:, :, random_x1: random_x1 + img_w] = data_dict["grasp_masks"]["wid"]
                    data_dict["bboxes"][:, [0, 2]] += random_x1
            elif self.mode in ["test", "val"]:
                pad_img[0: img_h, 0: img_w, :] = data_dict["rgb"]
                pad_depth[0: img_h, 0: img_w] = data_dict["depth"]
                pad_ins_masks[:, 0: img_h, 0: img_w] = data_dict["ins_masks"]
                pad_qua_masks[:, 0: img_h, 0: img_w] = data_dict["grasp_masks"]["qua"]
                pad_ang_masks[:, 0: img_h, 0: img_w] = data_dict["grasp_masks"]["ang"]
                pad_wid_masks[:, 0: img_h, 0: img_w] = data_dict["grasp_masks"]["wid"]


            data_dict["rgb"] = pad_img
            data_dict["depth"] = pad_depth
            data_dict["ins_masks"] = pad_ins_masks
            data_dict["grasp_masks"]["qua"] = pad_qua_masks
            data_dict["grasp_masks"]["ang"] = pad_ang_masks
            data_dict["grasp_masks"]["wid"] = pad_wid_masks
    

    def _resize(self, data_dict):
        ori_size = data_dict["rgb"].shape[0]
        tgt_size = self.cfg.img_size
        scale = tgt_size / ori_size

        data_dict["rgb"] = cv2.resize(data_dict["rgb"], (tgt_size, tgt_size))
        data_dict["depth"] = cv2.resize(data_dict["depth"], (tgt_size, tgt_size))
        data_dict["ins_masks"] = cv2.resize(data_dict["ins_masks"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["ins_masks"].shape[0] > 1 else \
                                     np.expand_dims(cv2.resize(data_dict["ins_masks"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["grasp_masks"]["qua"] = cv2.resize(data_dict["grasp_masks"]["qua"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["grasp_masks"]["qua"].shape[0] > 1 else \
                                    np.expand_dims(cv2.resize(data_dict["grasp_masks"]["qua"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["grasp_masks"]["ang"] = cv2.resize(data_dict["grasp_masks"]["ang"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["grasp_masks"]["ang"].shape[0] > 1 else \
                                    np.expand_dims(cv2.resize(data_dict["grasp_masks"]["ang"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["grasp_masks"]["wid"] = cv2.resize(data_dict["grasp_masks"]["wid"].transpose((1,2,0)), (tgt_size, tgt_size)).transpose((2,0,1)) if data_dict["grasp_masks"]["wid"].shape[0] > 1 else \
                                    np.expand_dims(cv2.resize(data_dict["grasp_masks"]["wid"].transpose((1,2,0)), (tgt_size, tgt_size)), 0).transpose((2,0,1))
        data_dict["bboxes"][:, :4] *= scale

    
    def _normalize_boxes(self, data_dict):
        h, w = data_dict["rgb"].shape[:2]
        data_dict["bboxes"][:, [0, 2]] /= w
        data_dict["bboxes"][:, [1, 3]] /= h
    
    def _normalize_img(self, data_dict):
        img = data_dict["rgb"] / 255.
        # img = (img - self.mean) / self.std
        img = img[:, :, (2,1,0)]
        img = np.transpose(img, (2, 0, 1))
        data_dict["rgb"] = img



    def __call__(self, data_dict):
        if self.mode == "train":
            self._photometric_distort(data_dict)
            self._random_mirror(data_dict)
        self._pad_to_square(data_dict)
        self._resize(data_dict)
        self._normalize_boxes(data_dict)
        self._normalize_img(data_dict)
```

## 2.box_utils.py

```
# -*- coding: utf-8 -*-
import torch
from itertools import product
from math import sqrt
import numpy as np


def box_iou(box_a, box_b):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    (n, A), B = box_a.shape[:2], box_b.shape[1]
    # add a dimension
    box_a = box_a[:, :, None, :].expand(n, A, B, 4)
    box_b = box_b[:, None, :, :].expand(n, A, B, 4)

    max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
    min_xy = torch.max(box_a[..., :2], box_b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    out = inter_area / (area_a + area_b - inter_area)
    return out if use_batch else out.squeeze(0)


def box_iou_numpy(box_a, box_b):
    (n, A), B = box_a.shape[:2], box_b.shape[1]
    # add a dimension
    box_a = np.tile(box_a[:, :, None, :], (1, 1, B, 1))
    box_b = np.tile(box_b[:, None, :, :], (1, A, 1, 1))

    max_xy = np.minimum(box_a[..., 2:], box_b[..., 2:])
    min_xy = np.maximum(box_a[..., :2], box_b[..., :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=100000)
    inter_area = inter[..., 0] * inter[..., 1]

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    return inter_area / (area_a + area_b - inter_area)


def match(cfg, box_gt, anchors, class_gt):
    # Convert prior boxes to the form of [xmin, ymin, xmax, ymax].
    decoded_priors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)

    overlaps = box_iou(box_gt, decoded_priors)  # (num_gts, num_achors)

    _, gt_max_i = overlaps.max(1)  # (num_gts, ) the max IoU for each gt box
    each_anchor_max, anchor_max_i = overlaps.max(0)  # (num_achors, ) the max IoU for each anchor

    # For the max IoU anchor for each gt box, set its IoU to 2. This ensures that it won't be filtered
    # in the threshold step even if the IoU is under the negative threshold. This is because that we want
    # at least one anchor to match with each gt box or else we'd be wasting training data.
    each_anchor_max.index_fill_(0, gt_max_i, 2)

    # Set the index of the pair (anchor, gt) we set the overlap for above.
    for j in range(gt_max_i.size(0)):
        anchor_max_i[gt_max_i[j]] = j

    anchor_max_gt = box_gt[anchor_max_i]  # (num_achors, 4)
    # For OCDI dataset
    conf = class_gt[anchor_max_i]  # the class of the max IoU gt box for each anchor
    # Others
    # conf = class_gt[anchor_max_i] + 1  # the class of the max IoU gt box for each anchor
    conf[each_anchor_max < cfg.pos_iou_thre] = -1  # label as neutral
    conf[each_anchor_max < cfg.neg_iou_thre] = 0  # label as background

    offsets = encode(anchor_max_gt, anchors)

    return offsets, conf, anchor_max_gt, anchor_max_i


def make_anchors(cfg, conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg.aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / cfg.img_size
            h = scale / ar / cfg.img_size

            prior_data += [x, y, w, h]

    return prior_data


def encode(matched, priors):
    variances = [0.1, 0.2]

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # 10 * (Xg - Xa) / Wa
    g_cxcy /= (variances[0] * priors[:, 2:])  # 10 * (Yg - Ya) / Ha
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 5 * log(Wg / Wa)
    g_wh = torch.log(g_wh) / variances[1]  # 5 * log(Hg / Ha)
    # return target for smooth_l1_loss

    offsets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]

    return offsets


def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def sanitize_coordinates_numpy(_x1, _x2, img_size, padding=0):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=1000000)
    x2 = np.clip(x2 + padding, a_min=0, a_max=img_size)

    return x1, x2


def crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def ones_crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    # ones = torch.ones_like(masks).float().cuda()
    ones = torch.ones_like(masks).float().to(masks.device)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    out_mask = ~crop_mask

    return masks * crop_mask.float() + ones * out_mask.float()



def crop_numpy(masks, boxes, padding=1):
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates_numpy(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates_numpy(boxes[:, 1], boxes[:, 3], h, padding)

    rows = np.tile(np.arange(w)[None, :, None], (h, 1, n))
    cols = np.tile(np.arange(h)[:, None, None], (1, w, n))

    masks_left = rows >= (x1.reshape(1, 1, -1))
    masks_right = rows < (x2.reshape(1, 1, -1))
    masks_up = cols >= (y1.reshape(1, 1, -1))
    masks_down = cols < (y2.reshape(1, 1, -1))

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask


def mask_iou(mask1, mask2):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).reshape(1, -1)
    area2 = torch.sum(mask2, dim=1).reshape(1, -1)
    union = (area1.t() + area2) - intersection
    ret = intersection / union

    return ret.cpu()
```

## 3.config.py

```
# -----------------------------------------------------------------------------
# Functions for parsing args
# -----------------------------------------------------------------------------
import copy
import os
from ast import literal_eval

import yaml


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg, cfg_list):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(value, cfg[subkey], subkey,
                                                 full_key)
        setattr(new_cfg, subkey, value)

    return new_cfg


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(original_type, replacement_type, original,
                         replacement, full_key))

```

## 4.dataset.py

```
import os
from typing import List, Union
import json
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from copy import deepcopy
import functools
from skimage.measure import regionprops
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.filters import gaussian
import matplotlib.pyplot as plt

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
from .augmentation import DataAugmentor

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'cocostuff': {
        "train": 965042,
        'val': 42095,
        'val-test': 42095
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)



class RefOCIDGraspDataset(Dataset):
    def __init__(self, root_path, input_size, word_length, mode="train"):
        super().__init__()
        json_path = os.path.join(root_path, f"{mode}_expressions.json")
        with open(json_path, "r") as f:
            self.meta_data = json.load(f)
        
        self.root_path = root_path
        self.keys = list(self.meta_data.keys())
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mode = mode
        
        self.cls_names = cls_names
        self.colors = colors
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        
        self.target_save_dir = os.path.join("./", mode)
        os.makedirs(self.target_save_dir, exist_ok=True)

        
    def __len__(self):
        return len(self.keys)
    

    def visualization(self, rgb, depth, masks, rects, grasp_masks, bbox, obj_cls, sentence):
        cls_list = list(self.cls_names.keys())

        print(rgb.shape, depth.shape, len(rects), masks.shape)
        mask_color = self.colors[str(obj_cls)]
        rgb_with_grasp = deepcopy(rgb)

        color_masks = np.repeat(masks[:, :, np.newaxis], 3, axis=-1) * mask_color
        rgb = (rgb * 255).astype(np.uint8)
        img_fuse = (rgb * 0.3 + color_masks * 0.7).astype(np.uint8)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_fuse, (x1, y1), (x2, y2), (255, 0, 0), 2)


        print(f'\nimg shape: {rgb.shape}')
        print('----------------boxes----------------')
        for rect in rects:
            print(rect)
        print('----------------labels---------------')
        print([cls_list[int(i)] for i in [obj_cls]], '\n')

        for rect in rects:
            name = cls_list[int(obj_cls)]
            color = self.colors[str(obj_cls)]
            center_x, center_y, width, height, theta, cls_id = rect
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.int0(box)
            cv2.drawContours(rgb_with_grasp, [box], 0, color.tolist(), 2)
        

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(img_fuse)
        ax.set_title('Masks & Bboxes')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(rgb_with_grasp)
        ax.set_title('Grasps')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(grasp_masks[0], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(grasp_masks[1], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(grasp_masks[2], cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
        ax.set_title('Angle')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(grasp_masks[3], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Width')
        ax.axis('off')
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        plt.savefig("./visualization.png")

    # @functools.lru_cache(maxsize=None)
    def _load_bbox(self, bbox):
        bbox = bbox.replace("[", "").replace("]", "").split(",")
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        return [x, y, w, h]

    # @functools.lru_cache(maxsize=None)
    def _load_rgb(self, path):
        image = cv2.imread(os.path.join(self.root_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        return image
    
    # @functools.lru_cache(maxsize=None)
    def _load_depth(self, path, factor=1000.):
        depth = cv2.imread(os.path.join(self.root_path, path), cv2.IMREAD_UNCHANGED) / factor
        depth = 1 - (depth / np.max(depth))

        return depth
    
    # @functools.lru_cache(maxsize=None)
    def _load_annos(self, path, target_cls):
        cls_annos_path = os.path.join(os.path.join(self.root_path, path), str(target_cls))
        file_id = cls_annos_path.split("/")[-2]
        cls_annos_path = os.path.join(cls_annos_path, f"{file_id}.txt")

        grasps_list = []
        with open(cls_annos_path, 'r') as f:
            points_list = []
            for count, line in enumerate(f):
                line = line.rstrip()
                [x, y] = line.split(' ')

                x = float(x)
                y = float(y)

                pt = (x, y)
                points_list.append(pt)

                if len(points_list) == 4:
                    p1, p2, p3, p4 = points_list
                    center_x = (p1[0] + p3[0]) / 2
                    center_y = (p1[1] + p3[1]) / 2
                    width  = np.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
                    height = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                    
                    # @NOTE
                    # Along x+ is 0 degree, increase by rotating anti-clockwise
                    # If you want to use opencv boxPoints & drawContours to visualize grasps
                    # Remember to take -theta as param :-)
                    theta = np.arctan2(p4[0] - p1[0], p4[1] - p1[1]) * 180 / np.pi
                    if theta > 0:
                        theta = theta-90
                    else:
                        theta = theta+90


                    grasps_list.append([center_x, center_y, width, height, theta, int(target_cls)])
                    points_list = []

        return grasps_list

    # @functools.lru_cache(maxsize=None)
    def _load_mask(self, path, target_cls):
        masks = cv2.imread(os.path.join(self.root_path, path), cv2.IMREAD_UNCHANGED)
        if target_cls == -1:
            # for loading instance masks
            return masks
        else:
            # For loading target semantic masks
            target_masks = (masks == target_cls)
            return target_masks

    def _match_masks_with_ref(self, bbox, ins_masks, masks):
        # Preparing bounding box for calculating IoU
        x1, y1, x2, y2 = bbox
        w, h = (x2-x1), (y2-y1)
        vertices = [[x1, y1], [x1+w, y1], [x2, y2], [x1, y1+h]]
        poly1 = Polygon(vertices)

        # Keep only the instance masks with correct class label
        ins_masks = ins_masks * masks
        ins_props = regionprops(ins_masks)

        max_iou = 0.0
        ins_idx = 0

        for ins in ins_props:
            _x1, _y1, _x2, _y2 = ins.bbox[1], ins.bbox[0], ins.bbox[3], ins.bbox[2]
            _w, _h = (_x2-_x1), (_y2-_y1)
            ins_vertices = [[_x1, _y1], [_x1+_w, _y1], [_x2, _y2], [_x1, _y1+_h]]
            poly2 = Polygon(ins_vertices)

            if poly1.intersects(poly2): 
                intersect = poly1.intersection(poly2).area
                union = poly1.union(poly2).area
                iou = intersect/union

                if iou > max_iou:
                    max_iou = iou
                    ins_idx = ins.label
        
        ins_masks = (ins_masks == ins_idx)

        return ins_masks

    def _match_grasps_with_ref(self, rects, ins_masks):
        # Check if the center of grasp falls in the target instance mask
        grasps = []
        for rect in rects:
            c_x, c_y = int(rect[0]), int(rect[1])
            if ins_masks[c_y, c_x]:
                grasps.append(rect)
        
        return grasps

    def _filter_grasps(self, rects):
        angles = []
        for rect in rects:
            angles.append(rect[4])
            
    
    def _generate_grasp_masks(self, grasps, width, height):
        pos_out = np.zeros((height, width))
        ang_out = np.zeros((height, width))
        wid_out = np.zeros((height, width))
        for rect in grasps:
            center_x, center_y, w_rect, h_rect, theta, cls_id = rect
            width_factor = float(100)

            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.int0(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < height
            cc = cc[mask_cc]
            rr = rr[mask_cc]


            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, width_factor) / width_factor
        
        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)
        
        return [pos_out, qua_out, ang_out, wid_out]
    
    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def convert(self, img, mask=None, grasp_quality_mask=None, grasp_sin_masks=None, grasp_cos_masks=None, grasp_width_mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        
        if grasp_quality_mask is not None:
            grasp_quality_mask = torch.from_numpy(grasp_quality_mask)
            if not isinstance(grasp_quality_mask, torch.FloatTensor):
                grasp_quality_mask = grasp_quality_mask.float()

        if grasp_sin_masks is not None:
            grasp_sin_masks = torch.from_numpy(grasp_sin_masks)
            if not isinstance(grasp_sin_masks, torch.FloatTensor):
                grasp_sin_masks = grasp_sin_masks.float()
        
        if grasp_cos_masks is not None:
            grasp_cos_masks = torch.from_numpy(grasp_cos_masks)
            if not isinstance(grasp_cos_masks, torch.FloatTensor):
                grasp_cos_masks = grasp_cos_masks.float()

        if grasp_width_mask is not None:
            grasp_width_mask = torch.from_numpy(grasp_width_mask)
            if not isinstance(grasp_width_mask, torch.FloatTensor):
                grasp_width_mask = grasp_width_mask.float()
        return img, mask, grasp_quality_mask, grasp_sin_masks, grasp_cos_masks, grasp_width_mask


    def __getitem__(self, index):
        data_dict = {}

        key = self.keys[index]
        ref_data = self.meta_data[key]
        obj_cls = int(self.cls_names[ref_data["class"]])

        # Get path to other data
        scene_path = ref_data["scene_path"]
        depth_path = scene_path.replace("rgb", "depth")
        annos_path = scene_path.replace("rgb", "Annotations_per_class")[:-4]
        masks_path = scene_path.replace("rgb", "seg_mask_labeled_combi")
        ins_masks_path = scene_path.replace("rgb", "seg_mask_instances_combi")

        # Read data
        rgb = self._load_rgb(scene_path) # [0 - 255]
        depth = self._load_depth(depth_path)
        annos = self._load_annos(annos_path, obj_cls)
        masks = self._load_mask(masks_path, obj_cls)
        ins_masks = self._load_mask(ins_masks_path, -1)
        bbox = self._load_bbox(ref_data["bbox"])
        sentence = ref_data["sentence"]

        # ins_masks = ins_masks * masks

        ins_masks = (self._match_masks_with_ref(bbox, ins_masks, masks) * 255).astype(np.uint8)

        grasps = self._match_grasps_with_ref(annos, ins_masks)

        assert rgb.shape[:2] == depth.shape

        height, width = depth.shape

        grasp_masks = self._generate_grasp_masks(grasps, width, height)

        grasp_quality_masks = grasp_masks[1]
        grasp_angle_masks = grasp_masks[2]
        grasp_width_masks = grasp_masks[3]

        # Image transforms
        img_size = rgb.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, True)
        rgb = cv2.warpAffine(rgb,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_CUBIC,
                                borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            ins_masks = cv2.warpAffine(ins_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            grasp_quality_masks = cv2.warpAffine(grasp_quality_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            grasp_angle_masks = cv2.warpAffine(grasp_angle_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            grasp_width_masks = cv2.warpAffine(grasp_width_masks,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
            
            # Normalize and pre-process target masks
            ins_masks = ins_masks / 255.
            grasp_quality_masks = grasp_quality_masks / 255.
            grasp_angle_masks = grasp_angle_masks * np.pi / 180.
            grasp_width_masks = grasp_width_masks / 255.
            grasp_sin_masks = np.sin(2 * grasp_angle_masks)
            grasp_cos_masks = np.cos(2 * grasp_angle_masks)
            
            word_vec = tokenize(sentence, self.word_length, True).squeeze(0)

            # self.visualization(rgb, depth, ins_masks, grasps, (grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks), bbox, obj_cls, sentence)

            rgb, ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks = self.convert(
                rgb, ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks
            )

            # np.savez(
            #     os.path.join(self.target_save_dir, f"{key}.npz"),
            #     img=rgb.cpu().numpy(),
            #     word_vec=word_vec.cpu().numpy(),
            #     ins_masks=ins_masks.cpu().numpy(),
            #     grasp_quality_masks=grasp_quality_masks.cpu().numpy(),
            #     grasp_sin_masks=grasp_sin_masks.cpu().numpy(),
            #     grasp_cos_masks=grasp_cos_masks.cpu().numpy(),
            #     grasp_width_masks=grasp_width_masks.cpu().numpy(),
            #     grasps=grasps
            # )


            return rgb, (ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks), word_vec
        
        elif self.mode == 'val':
            word_vec = tokenize(sentence, self.word_length, True).squeeze(0)

            ins_masks = ins_masks / 255.
            grasp_quality_masks = grasp_quality_masks / 255.
            grasp_angle_masks = grasp_angle_masks * np.pi / 180.
            grasp_width_masks = grasp_width_masks / 255.
            grasp_sin_masks = np.sin(2 * grasp_angle_masks)
            grasp_cos_masks = np.cos(2 * grasp_angle_masks)

            rgb = self.convert(rgb)[0]
            params = {
                # 'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            # np.savez(
            #     os.path.join(self.target_save_dir, f"{key}.npz"),
            #     img=rgb.cpu().numpy(),
            #     word_vec=word_vec.cpu().numpy(),
            #     ins_masks=ins_masks,
            #     grasp_quality_masks=grasp_quality_masks,
            #     grasp_sin_masks=grasp_sin_masks,
            #     grasp_cos_masks=grasp_cos_masks,
            #     grasp_width_masks=grasp_width_masks,
            #     grasps=grasps,
            #     params=params
            # )
            return rgb, (ins_masks, grasp_quality_masks, grasp_sin_masks, grasp_cos_masks, grasp_width_masks), word_vec, params
        elif self.mode == 'test':
            rgb = self.convert(rgb)[0]
            word_vec = tokenize(sentence, self.word_length, True).squeeze(0)
            params = {
                # 'ori_img': ori_img,
                # 'seg_id': seg_id,
                # 'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sentence

            }
            # np.savez(
            #     os.path.join(self.target_save_dir, f"{key}.npz"),
            #     img=rgb.cpu().numpy(),
            #     word_vec=word_vec.cpu().numpy(),
            #     params=params
            # )
            return rgb, word_vec, params



class GraspTransforms:
    # Class for converting cv2-like rectangle formats and generate grasp-quality-angle-width masks

    def __init__(self, width_factor=100, width=640, height=480):
        self.width_factor = width_factor
        self.width = width 
        self.height = height

    def __call__(self, grasp_rectangles, target):
        # grasp_rectangles: (M, 4, 2)
        M = grasp_rectangles.shape[0]
        p1, p2, p3, p4 = np.split(grasp_rectangles, 4, axis=1)
        
        center_x = (p1[..., 0] + p3[..., 0]) / 2
        center_y = (p1[..., 1] + p3[..., 1]) / 2
        
        width  = np.sqrt((p1[..., 0] - p4[..., 0]) * (p1[..., 0] - p4[..., 0]) + (p1[..., 1] - p4[..., 1]) * (p1[..., 1] - p4[..., 1]))
        height = np.sqrt((p1[..., 0] - p2[..., 0]) * (p1[..., 0] - p2[..., 0]) + (p1[..., 1] - p2[..., 1]) * (p1[..., 1] - p2[..., 1]))
        
        theta = np.arctan2(p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1]) * 180 / np.pi
        theta = np.where(theta > 0, theta - 90, theta + 90)

        target = np.tile(np.array([[target]]), (M,1))

        return np.concatenate([center_x, center_y, width, height, theta, target], axis=1)

    def inverse(self, grasp_rectangles):
        boxes = []
        for rect in grasp_rectangles:
            center_x, center_y, width, height, theta = rect[:5]
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def generate_masks(self, grasp_rectangles):
        pos_out = np.zeros((self.height, self.width))
        ang_out = np.zeros((self.height, self.width))
        wid_out = np.zeros((self.height, self.width))
        for rect in grasp_rectangles:
            center_x, center_y, w_rect, h_rect, theta = rect[:5]
            
            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.intp(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < self.width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < self.height
            cc = cc[mask_cc]
            rr = rr[mask_cc]
            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, self.width_factor) / self.width_factor
        
        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)
        
        
        return {'pos': pos_out, 
                'qua': qua_out, 
                'ang': ang_out, 
                'wid': wid_out}



class OCIDVLGDataset(Dataset):
    
    """ OCID-Vision-Language-Grasping dataset with referring expressions and grasps """

    def __init__(self, 
                 root_dir,
                 split, 
                 transform_img = None,
                 transform_grasp = GraspTransforms(),
                 input_size = 416,
                 word_length = 20,
                 with_depth = True, 
                 with_segm_mask = True,
                 with_grasp_masks = True,
                 version="multiple"
    ):
        super(OCIDVLGDataset, self).__init__()
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, "data_split")
        self.split_map = {'train': 'train_expressions.json', 
                          'val': 'val_expressions.json',
                          'test': 'test_expressions.json'
                         }
        self.split = split
        self.refer_dir = os.path.join(root_dir, "refer", version)
        
        self.transform_img = transform_img
        self.transform_grasp = transform_grasp
        self.with_depth = with_depth
        self.with_segm_mask = with_segm_mask
        self.with_grasp_masks = with_grasp_masks
        # assert (self.transform_grasp and self.with_grasp_masks) or (not self.transform_grasp and not self.with_grasp_masks)

        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

        self._load_dicts()
        self._load_split()

    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
        cnames_inv = {int(v):k for k,v in cnames.items()}
        subnames_inv = {v:k for k,v in subnames.items()}
        self.class_names = cnames 
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = sub_to_class
        os.chdir(cwd)

    def _load_split(self):
        refer_data = json.load(open(os.path.join(self.refer_dir, self.split_map[self.split])))
        self.seq_paths, self.img_names, self.scene_ids = [], [], []
        self.bboxes, self.grasps = [], []
        self.sent_to_index, self.sent_indices = {}, []
        self.rgb_paths, self.depth_paths, self.mask_paths = [], [], []
        self.targets, self.sentences, self.semantics, self.objIDs = [], [], [], []
        n = 0
        for item in refer_data['data']:
            seq_path, im_name = item['image_filename'].split(',')
            self.seq_paths.append(seq_path)
            self.img_names.append(im_name)
            self.scene_ids.append(item['image_filename'])
            self.bboxes.append(item['box'])
            self.grasps.append(item['grasps'])
            self.objIDs.append(item['answer'])
            self.targets.append(item['target'])
            self.sentences.append(item['question'])
            self.semantics.append(item['program'])
            self.rgb_paths.append(os.path.join(seq_path, "rgb", im_name))
            self.depth_paths.append(os.path.join(seq_path, "depth", im_name))
            self.mask_paths.append(os.path.join(seq_path, "seg_mask_instances_combi", im_name))
            self.sent_indices.append(item['question_index'])
            self.sent_to_index[item['question_index']] = n
            n += 1
            
    def get_index_from_sent(self, sent_id):
        return self.sent_to_index[sent_id]

    def get_sent_from_index(self, n):
        return self.sent_indices[n]
    
    def _load_sent(self, sent_id):
        n = self.get_index_from_sent(sent_id)
        
        scene_id = self.scene_ids[n]
       
        img_path = os.path.join(self.root_dir, self.rgb_paths[n])
        img = self.get_image_from_path(img_path)
        
        x, y, w, h = self.bboxes[n]
        bbox = np.asarray([x, y, x+w, y+h])
        
        sent = self.sentences[n]
        
        target = self.targets[n]
        target_idx = self.class_instance_names[target]
        objID = self.objIDs[n]
        
        grasps = np.asarray(self.grasps[n])
        
        result = {'img': self.transform_img(img) if self.transform_img else img, 
                  'grasps':  self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'grasp_rects': self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
                  'sentence': sent,
                  'target': target,
                  'objID': objID,
                  'bbox': bbox,
                  'target_idx': target_idx,
                  'sent_id': sent_id,
                  'scene_id': scene_id,
                  'img_path': img_path
                 }
        
        if self.with_depth:
            depth_path = os.path.join(self.root_dir, self.depth_paths[n])
            depth = self.get_depth_from_path(depth_path)
            result = {**result, 'depth': torch.from_numpy(depth) if self.transform_img else depth}

        if self.with_segm_mask:
            mask_path = os.path.join(self.root_dir, self.mask_paths[n])
            msk_full = self.get_mask_from_path(mask_path)
            msk = np.where(msk_full == objID, True, False)
            result = {**result, 'mask': torch.from_numpy(msk) if self.transform_img else msk}

        if self.with_grasp_masks:
            grasp_masks = self.transform_grasp.generate_masks(result['grasp_rects'])
            result = {**result, 'grasp_masks': grasp_masks}
        
        result = self.preprocess(result)
        
        return result
    
    def get_transform_mat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def preprocess(self, data):
        img = data["img"]
        sent = data["sentence"]
        if np.max(data["mask"]) <= 1.0:
            ins_mask = (data["mask"] * 255).astype(np.uint8)
        else:
            ins_mask = data["mask"]
        
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_ang_mask = data["grasp_masks"]["ang"]
        grasp_wid_mask = data["grasp_masks"]["wid"]

        img_size = img.shape[:2]
        mat, mat_inv = self.get_transform_mat(img_size, True)

        img = cv2.warpAffine(
            img, mat, self.input_size, flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )

        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)

        ins_mask = cv2.warpAffine(ins_mask,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
        grasp_qua_mask = cv2.warpAffine(grasp_qua_mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)
        
        grasp_ang_mask = cv2.warpAffine(grasp_ang_mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)
        
        grasp_wid_mask = cv2.warpAffine(grasp_wid_mask,
                                mat,
                                self.input_size,
                                flags=cv2.INTER_LINEAR,
                                borderValue=0.)


        ins_mask = ins_mask / 255.
        grasp_qua_mask = grasp_qua_mask / 255.
        grasp_ang_mask = grasp_ang_mask * np.pi / 180.
        grasp_wid_mask = grasp_wid_mask / 255.
        grasp_sin_mask = np.sin(2 * grasp_ang_mask)
        grasp_cos_mask = np.cos(2 * grasp_ang_mask)

        word_vec = tokenize(sent, self.word_length, True).squeeze(0)

        data["img"] = img
        data["mask"] = ins_mask
        data["grasp_masks"]["qua"] = grasp_qua_mask
        data["grasp_masks"]["ang"] = grasp_ang_mask
        data["grasp_masks"]["wid"] = grasp_wid_mask
        data["grasp_masks"]["sin"] = grasp_sin_mask
        data["grasp_masks"]["cos"] = grasp_cos_mask
        data["word_vec"] = word_vec
        data["inverse"] = mat_inv
        data["ori_size"] = np.array(img_size)
        
        # del data["sentence"]
        
        return data

    def __len__(self):
        return len(self.sent_indices)
    
    def __getitem__(self, n):
        sent_id = self.get_sent_from_index(n)
        data = self._load_sent(sent_id)
        
        return data
    
    @staticmethod
    def transform_grasp_inv(grasp_pt):
        pass
    
    # @functools.lru_cache(maxsize=None)
    def get_image_from_path(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(os.path.exists(path))
            print(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img

    # @functools.lru_cache(maxsize=None)
    def get_mask_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # @functools.lru_cache(maxsize=None)
    def get_depth_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000. # mm -> m
    
    def get_image(self, n):
        img_path = os.path.join(self.root_dir, self.imgs[n])
        return self.get_image_from_path(img_path)
    
    def get_annotated_image(self, n, text=True):
        sample = self.__getitem__(n)
        
        img, sent, grasps, bbox = sample['img'], sample['sentence'], sample['grasp_rects'], sample['bbox']
        if isinstance(img, torch.FloatTensor):
            img = img.permute(1,2,0)
            img = (img.cpu().numpy() * 255).astype(np.uint8)
        if self.transform_img:
            img = np.asarray(tfn.to_pil_image(img))
        if self.transform_grasp:
            #grasps = list(map(self.transform_grasp_inv, list(grasps)))
            grasps = self.transform_grasp.inverse(grasps)

        tmp = img.copy()
        for entry in grasps:
            ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
            tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
            tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
        
        tmp = cv2.rectangle(tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
        if text:
            tmp = cv2.putText(tmp, sent, (0,10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
        return tmp

    def visualization(self, n, save_path):
        s = self.__getitem__(n)

        rgb = s['img']
        if isinstance(rgb, torch.FloatTensor):
            rgb = rgb.permute(1,2,0)
            rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        depth = (0xff * s['depth'] / 3).astype(np.uint8)
        ii = self.get_annotated_image(n, text=False)
        sentence = s['sentence']
        msk = s['mask'].astype(np.uint8) / 255
        # msk_img = (rgb * 0.3).astype(np.uint8).copy()
        # msk_img[msk, 0] = 255

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(msk)
        ax.set_title('Segm Mask')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(ii)
        ax.set_title('Box & Grasp')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(s['grasp_masks']['qua'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Grasp quality')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(s['grasp_masks']['sin'], cmap='rainbow', vmin=-1, vmax=1)
        ax.set_title('Angle-cosine')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(s['grasp_masks']['cos'], cmap='rainbow', vmin=-1, vmax=1)
        ax.set_title('Angle-sine')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(s['grasp_masks']['wid'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Width')
        ax.axis('off')
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        print("save")
        plt.savefig(os.path.join(save_path, f"sample_{n}.png"))
    
    @staticmethod
    def collate_fn(batch):
        return {
            "img": torch.stack([x["img"] for x in batch]),
            "depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch]),
            "mask": torch.stack([torch.from_numpy(x["mask"]).float() for x in batch]),
            "grasp_masks" : {
                "qua": torch.stack([torch.from_numpy(x["grasp_masks"]["qua"]).float() for x in batch]),
                "sin": torch.stack([torch.from_numpy(x["grasp_masks"]["sin"]).float() for x in batch]),
                "cos": torch.stack([torch.from_numpy(x["grasp_masks"]["cos"]).float() for x in batch]),
                "wid": torch.stack([torch.from_numpy(x["grasp_masks"]["wid"]).float() for x in batch])
            },
            "word_vec": torch.stack([x["word_vec"].long() for x in batch]),
            "grasps": [x["grasps"] for x in batch],
            "target": [x["target"] for x in batch],
            "sentence": [x["sentence"] for x in batch],
            "bbox": [x["bbox"] for x in batch],
            "target_idx": [x["target_idx"] for x in batch],
            "sent_id": [x["sent_id"] for x in batch],
            "scene_id": [x["scene_id"] for x in batch],
            "inverse": [x["inverse"] for x in batch],
            "ori_size": [x["ori_size"] for x in batch],
            "img_path": [x["img_path"] for x in batch]
        }
            
        


class OCIDGraspDataset(Dataset):
    
    """ OCID-Grasp dataset """

    def __init__(self, 
                 cfg,
                 split):
        self.cfg = cfg
        self.split = split
        self.root_dir = cfg.root_dir
        self.img_size = cfg.img_size
        self.depth_factor = cfg.depth_factor
        self.with_grasp_masks = cfg.with_grasp_masks
        self.with_sem_masks = cfg.with_sem_masks
        self.with_ins_masks = cfg.with_ins_masks
        self.with_depth = cfg.with_depth
        self.grasp_transforms = GraspTransforms()

        aug_mode = "train" if self.split == "training_0" else "test"
        self.data_augmentor = DataAugmentor(cfg, mode=aug_mode)
        # self.data_augmentor = DataAugmentor(cfg)
        
        self._load_dicts()
        self.num_classes = len(cnames)

        with open(os.path.join(cfg.root_dir, "data_split", split + ".txt"), "r") as fid:
            self.meta = [x.strip().split(',') for x in fid.readlines()]


    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
        cnames_inv = {int(v):k for k,v in cnames.items()}
        subnames_inv = {v:k for k,v in subnames.items()}
        self.class_names = cnames 
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = sub_to_class
        os.chdir(cwd)

    
    def _get_rgb_image(self, scene_id, img_f, data_dict):
        img_path = os.path.join(self.root_dir, scene_id, "rgb", img_f)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        data_dict["rgb"] = img

    
    def _get_depth_image(self, scene_id, img_f, data_dict):
        depth_path = os.path.join(self.root_dir, scene_id, "depth", img_f)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / float(self.depth_factor)
        depth = 1 - (depth / np.max(depth))
        data_dict["depth"] = depth
        
    
    def _get_sem_mask(self, scene_id, img_f, data_dict):
        sem_mask = cv2.imread(os.path.join(self.root_dir, scene_id, "seg_mask_labeled_combi", img_f), cv2.IMREAD_UNCHANGED)
        data_dict["sem_mask"] = sem_mask
        return sem_mask

    
    def _get_ins_mask(self, scene_id, img_f, data_dict):
        # Load semantic mask first if the with_sem_masks set to False
        if "sem_mask" not in data_dict.keys():
            sem_mask = cv2.imread(os.path.join(self.root_dir, scene_id, "seg_mask_labeled_combi", img_f), cv2.IMREAD_UNCHANGED)
        else:
            sem_mask = data_dict["sem_mask"]
        ins_mask = cv2.imread(os.path.join(self.root_dir, scene_id, "seg_mask_instances_combi", img_f), cv2.IMREAD_UNCHANGED)

        labels     = []
        bboxes     = []
        ins_masks  = []

        props = regionprops(sem_mask)
        for prop in props:
            cls_id = prop.label
            
            # Get binary mask for each semantic class
            bin_mask = (sem_mask == cls_id).astype('int8')
            # Get corresponding semantic mask (may contains multiple instances)
            cls_ins_mask = (ins_mask * bin_mask)

            # Get regions for each instance
            ins_props = regionprops(cls_ins_mask)
            for ins in ins_props:
                labels.append(cls_id)
                bboxes.append([ins.bbox[1], ins.bbox[0], ins.bbox[3], ins.bbox[2], cls_id])
                mask = (cls_ins_mask == ins.label).astype('int8').astype('float32')
                ins_masks.append(mask)
        
        bboxes = np.array(bboxes).astype('float32')
        labels = np.array(labels)
        ins_masks  = np.array(ins_masks)

        data_dict["bboxes"] = bboxes
        data_dict["labels"] = labels
        data_dict["ins_masks"] = ins_masks


    
    def _get_per_cls_grasp_rects(self, scene_id, img_f, data_dict):
        anno_path = os.path.join(self.root_dir, scene_id, "Annotations_per_class", img_f[:-4])
        grasps_list = []
        for cls_id in os.listdir(anno_path):
            grasp_path = os.path.join(anno_path, cls_id, img_f[:-4]+".txt")
            with open(grasp_path, 'r') as f:
                points_list = []
                for count, line in enumerate(f):
                    line = line.rstrip()
                    [x, y] = line.split(' ')

                    x = float(x)
                    y = float(y)

                    pt = (x, y)
                    points_list.append(pt)

                    if len(points_list) == 4:
                        p1, p2, p3, p4 = points_list
                        center_x = (p1[0] + p3[0]) / 2
                        center_y = (p1[1] + p3[1]) / 2
                        width  = np.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
                        height = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                        
                        # @NOTE
                        # Along x+ is 0 degree, increase by rotating anti-clockwise
                        # If you want to use opencv boxPoints & drawContours to visualize grasps
                        # Remember to take -theta as param :-)
                        theta = np.arctan2(p4[0] - p1[0], p4[1] - p1[1]) * 180 / np.pi
                        if theta > 0:
                            theta = theta-90
                        else:
                            theta = theta+90


                        grasps_list.append([center_x, center_y, width, height, theta, int(cls_id)])
                        points_list = []
        data_dict["raw_grasp_rects"] = grasps_list



    def _get_grasp_mask(self, scene_id, img_f, data_dict):
        grasp_rects = data_dict["raw_grasp_rects"]
        data_dict["grasp_masks"] = {}
        bboxes = data_dict["bboxes"]
        labels = data_dict["labels"]
        masks = data_dict["ins_masks"] # @NOTE there may exist instance with no grasp annotations, thus we need to filter them out
        assert bboxes.shape[0] == masks.shape[0], "inconsistent bounding boxes and instances, check the data"
        num_ins = bboxes.shape[0]

        ins_grasp_rects = []
        ins_grasp_masks = []
        ins_bboxes = []
        ins_masks = []
        ins_labels = []

        for i in range(num_ins):
            box = bboxes[i]
            mask = masks[i]
            label = labels[i]
            tmp = []
            for rect in grasp_rects:
                center_x, center_y, w, h = rect[:4]
                cls_id = rect[-1]
                # Grasp rect and bbox should have the same cls_id
                if int(cls_id) == int(box[4]):
                    # Center of grasp rect in bbox
                    if mask[int(center_y), int(center_x)]:
                        tmp.append(rect)
            if len(tmp) > 0:
                ins_grasp_masks.append(self.grasp_transforms.generate_masks(tmp))
                ins_grasp_rects.append(tmp)
                ins_bboxes.append(box)
                ins_masks.append(mask)
                ins_labels.append(label)

        data_dict["bboxes"] = np.asarray(ins_bboxes)
        data_dict["labels"] = np.asarray(ins_labels)
        data_dict["ins_masks"] = np.asarray(ins_masks)
        data_dict["ins_grasp_rects"] = ins_grasp_rects
        data_dict["grasp_masks"]["qua"] = np.asarray([gm["qua"] / 255 for gm in ins_grasp_masks])
        data_dict["grasp_masks"]["ang"] = np.asarray([gm["ang"] for gm in ins_grasp_masks])
        data_dict["grasp_masks"]["wid"] = np.asarray([gm["wid"] / 255 for gm in ins_grasp_masks])


    def __len__(self):
        return len(self.meta)

    
    def __getitem__(self, index):
        data_dict = {}
        scene_id, img_f = self.meta[index]
        data_dict["scene_id"] = scene_id
        data_dict["img_f"] = img_f

        img_path = os.path.join(self.root_dir, scene_id, "rgb", img_f)
        img = cv2.imread(img_path)
        data_dict["rgb"] = img
        data_dict["ori_size"] = img.shape[:2]

        if self.with_depth:
            self._get_depth_image(scene_id, img_f, data_dict)
        if self.with_sem_masks:
            self._get_sem_mask(scene_id, img_f, data_dict)
        if self.with_ins_masks:
            self._get_ins_mask(scene_id, img_f, data_dict)
        if self.with_grasp_masks:
            self._get_per_cls_grasp_rects(scene_id, img_f, data_dict)
            self._get_grasp_mask(scene_id, img_f, data_dict)
        
        
        self.data_augmentor(data_dict)

        data_dict["grasp_masks"]["sin"] = np.sin(2 * data_dict["grasp_masks"]["ang"])
        data_dict["grasp_masks"]["cos"] = np.cos(2 * data_dict["grasp_masks"]["ang"])

        return data_dict


    def visualization(self, index, tgt_dir, with_preprocessing=False):
        data_dict = {}
        scene_id, img_f = self.meta[index]
        data_dict["scene_id"] = scene_id
        data_dict["img_f"] = img_f

        img_path = os.path.join(self.root_dir, scene_id, "rgb", img_f)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        data_dict["rgb"] = img

        if self.with_depth:
            self._get_depth_image(scene_id, img_f, data_dict)
        if self.with_sem_masks:
            self._get_sem_mask(scene_id, img_f, data_dict)
        if self.with_ins_masks:
            self._get_ins_mask(scene_id, img_f, data_dict)
        if self.with_grasp_masks:
            self._get_per_cls_grasp_rects(scene_id, img_f, data_dict)
            self._get_grasp_mask(scene_id, img_f, data_dict)
        if with_preprocessing:
            self.data_augmentor(data_dict)
            img = data_dict["rgb"].transpose((1,2,0))
        # img = img / 255.

        num_ins = data_dict["bboxes"].shape[0]
        
        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(np.clip(img[:, :, ::-1], 0.0, 1.0)*255)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(data_dict["depth"], cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(data_dict["sem_mask"])
        ax.set_title('Segm Mask')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(tgt_dir, f"raw-data.png"))
        plt.clf()
        plt.cla()
        plt.close()


        for i in range(num_ins):
            fig = plt.figure(figsize=(20, 2))

            ins_box = data_dict["bboxes"][i]
            ins_mask = data_dict["ins_masks"][i]
            ins_label = self.idx_to_class[data_dict["labels"][i]]
            ins_grasp_rects = data_dict["ins_grasp_rects"][i]
            ins_grasp_qua_masks = data_dict["grasp_masks"]["qua"][i]
            ins_grasp_ang_masks = data_dict["grasp_masks"]["ang"][i]
            ins_grasp_wid_masks = data_dict["grasp_masks"]["wid"][i]

            grasps = self.grasp_transforms.inverse(ins_grasp_rects)
            tmp = img.copy()
            h, w, c = tmp.shape
            for entry in grasps:
                ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
                tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
                tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
                tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
                tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
            if ins_box[0] <= 1:
                tmp = cv2.rectangle(tmp, (int(ins_box[0]*w),int(ins_box[1]*h)), (int(ins_box[2]*w),int(ins_box[3]*h)), (0,0,0xff), 2)
            else:
                tmp = cv2.rectangle(tmp, (int(ins_box[0]),int(ins_box[1])), (int(ins_box[2]),int(ins_box[3])), (0,0,0xff), 2)

            ax = fig.add_subplot(1, 5, 1)
            ax.imshow(tmp[:, :, ::-1])
            ax.set_title('Bboxes & Grasps')
            ax.axis('off')

            ax = fig.add_subplot(1, 5, 2)
            tmp_mask = np.expand_dims(ins_mask, axis=-1).repeat(3, axis=-1)
            ax.imshow(tmp_mask*0.6 + tmp[:, :, ::-1] *0.4)
            ax.set_title('ins mask')
            ax.axis('off')

            ax = fig.add_subplot(1, 5, 3)
            plot = ax.imshow(ins_grasp_qua_masks, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Grasp quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(1, 5, 4)
            plot = ax.imshow(ins_grasp_ang_masks, cmap='rainbow', vmin=-1, vmax=1)
            ax.set_title('Grasp angle')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(1, 5, 5)
            plot = ax.imshow(ins_grasp_wid_masks, cmap='rainbow', vmin=0, vmax=1)
            ax.set_title('Grasp width')
            ax.axis('off')
            plt.colorbar(plot)

            plt.tight_layout()
            plt.savefig(os.path.join(tgt_dir, f"ins-{i}-{ins_label}.png"))


    @staticmethod
    def collate_fn(batch):
        return {
            "scene_id": [x["scene_id"] for x in batch],
            "img_f": [x["img_f"] for x in batch],
            "ori_size": batch[0]["ori_size"],
            "rgb": torch.stack([torch.from_numpy(x["rgb"]) for x in batch]),
            "depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch]).unsqueeze(1),
            "labels": [torch.from_numpy(x["labels"]).long() for x in batch],
            "bboxes": [torch.from_numpy(x["bboxes"]) for x in batch],
            "ins_masks": [torch.from_numpy(x["ins_masks"]).float() for x in batch],
            "sem_mask": torch.stack([torch.from_numpy(x["sem_mask"]).float() for x in batch]),
            "grasp_rects": [x["ins_grasp_rects"] for x in batch],
            "grasp_masks" : {
                "qua": [torch.from_numpy(x["grasp_masks"]["qua"]).float() for x in batch],
                "sin": [torch.from_numpy(x["grasp_masks"]["sin"]).float() for x in batch],
                "cos": [torch.from_numpy(x["grasp_masks"]["cos"]).float() for x in batch],
                "wid": [torch.from_numpy(x["grasp_masks"]["wid"]).float() for x in batch]
            },
        }
            
```

## 5.grasp_eval.py

```
import cv2
import numpy as np
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from .box_utils import crop, box_iou


@torch.no_grad()
def draw_lincomb(ids_p, proto_data, masks, img_name, target_dir=None):
    for kdx in range(masks.shape[0]):
        target_id = int(ids_p[kdx])
        # jdx = kdx + -1
        coeffs = masks[kdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        p_h, p_w, _ = proto_data.size()
        arr_img = np.zeros([p_h * arr_h, p_w * arr_w])
        arr_run = np.zeros([p_h * arr_h, p_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1 / (1 + np.exp(-running_total)))

                arr_img[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (proto_data[:, :, idx[i]] / torch.max(
                    proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (running_total_nonlin > 0.5).astype(np.float)

        arr_img = ((arr_img + 1) * 127.5).astype('uint8')
        arr_img = cv2.applyColorMap(arr_img, cv2.COLORMAP_WINTER)
        if target_dir is None:
            cv2.imwrite(f'results/ocid/lincomb_{img_name}', arr_img)
        else:
            if not os.path.exists(f'{target_dir}/{target_id}'):
                os.makedirs(f'{target_dir}/{target_id}')

            cv2.imwrite(f'{target_dir}/{target_id}/lincomb_{img_name}', arr_img)


@torch.no_grad()
def fast_nms(cfg, box_pred_kept, cls_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept):
    cls_pred_kept, idx = cls_pred_kept.sort(1, descending=True)

    idx = idx[:, :cfg.top_k]
    cls_pred_kept = cls_pred_kept[:, :cfg.top_k]
    num_classes, num_dets = idx.size()

    box_pred_kept = box_pred_kept[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)
    ins_coef_pred_kept = ins_coef_pred_kept[idx.reshape(-1), :].reshape(num_classes, num_dets, cfg.num_protos)
    grasp_coef_pred_kept = grasp_coef_pred_kept[idx.reshape(-1), :].reshape(num_classes, num_dets, 4, cfg.num_protos)

    # Calculate IoU between predicted boxes
    iou = box_iou(box_pred_kept, box_pred_kept)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_pred_kept.device)[:, None].expand_as(keep)
    class_ids = class_ids[keep]
    cls_pred_kept = cls_pred_kept[keep]
    box_pred_kept = box_pred_kept[keep]
    ins_coef_pred_kept = ins_coef_pred_kept[keep]
    grasp_coef_pred_kept = grasp_coef_pred_kept[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    cls_pred_kept, idx = cls_pred_kept.sort(0, descending=True)
    idx = idx[:cfg.max_detections]
    cls_pred_kept = cls_pred_kept[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_pred_kept = box_pred_kept[idx]
    ins_coef_pred_kept = ins_coef_pred_kept[idx]
    grasp_coef_pred_kept = grasp_coef_pred_kept[idx]

    
    return class_ids, cls_pred_kept, box_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept



# IMPORTANT
# For batch_size=1 only
@torch.no_grad()
def ssg_post_processing(cfg, output_dict, data_dict):
    ori_size = data_dict["ori_size"]
    ori_h, ori_w = ori_size
    input_size = max(ori_h, ori_w)
    protos = output_dict["protos"].squeeze()
    cls_pred = output_dict["cls_pred"].squeeze()
    box_pred = output_dict["box_pred"].squeeze()
    ins_coef_pred = output_dict["ins_coef_pred"].squeeze()
    grasp_coef_pred = output_dict["grasp_coef_pred"].squeeze()
    anchors = torch.tensor(output_dict["anchors"], device=protos.device).reshape(-1, 4).squeeze()
    # print("anchors", anchors.shape)
    # print("cls_pred", cls_pred.shape)
    B, N = cls_pred.shape[:2]

    cls_pred = cls_pred.transpose(1, 0).contiguous()

    # Exclude the background class
    cls_pred = cls_pred[1:, :]
    # get the max score class of 19248 predicted boxes
    cls_pred_max, _ = torch.max(cls_pred, dim=0)
    # print("cls_pred_max", cls_pred_max.shape)

    # filter predicted boxes according the class score
    keep = (cls_pred_max > cfg.nms_score_thre) # [N_anchors]
    # print(anchors.shape, cls_pred.shape, box_pred.shape, ins_coef_pred.shape, grasp_coef_pred.shape)
    # print(keep.shape)
    anchors_kept = anchors[keep, :]
    cls_pred_kept = cls_pred[:, keep]
    box_pred_kept = box_pred[keep, :]
    ins_coef_pred_kept = ins_coef_pred[keep, :]
    grasp_coef_pred_kept = grasp_coef_pred[keep, :]

    # decode boxes
    box_pred_kept = torch.cat((anchors_kept[:, :2] + box_pred_kept[:, :2] * 0.1 * anchors_kept[:, 2:],
                          anchors_kept[:, 2:] * torch.exp(box_pred_kept[:, 2:] * 0.2)), 1)
    box_pred_kept[:, :2] -= box_pred_kept[:, 2:] / 2
    box_pred_kept[:, 2:] += box_pred_kept[:, :2]
    box_pred_kept = torch.clip(box_pred_kept, min=0., max=1.)

    # Fast NMS
    class_ids, cls_pred_kept, box_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept = fast_nms(cfg, box_pred_kept, cls_pred_kept, ins_coef_pred_kept, grasp_coef_pred_kept)

    keep = (cls_pred_kept > 0.3)
    if not keep.any():
        print("No valid instance")
    else:
        class_ids = class_ids[keep]
        cls_pred_kept = cls_pred_kept[keep]
        box_pred_kept = box_pred_kept[keep]
        ins_coef_pred_kept = ins_coef_pred_kept[keep]   
        grasp_coef_pred_kept = grasp_coef_pred_kept[keep]

    class_ids = (class_ids + 1)
    class_ids = class_ids.cpu().numpy()
    
    # if cfg.vis_protos:
    #     ones_coef = torch.ones(pos_coef_p.shape).float().cuda()
    #     # print("ProtoTypes")
    #     draw_lincomb(ids_p, proto_p, ones_coef, "prototypes.png", target_dir)

    #     # print("Semantic")
    #     draw_lincomb(ids_p, proto_p, coef_p, "cogr-sem.png", target_dir)
    #     # print("Grasp pos")
    #     draw_lincomb(ids_p, proto_p, pos_coef_p, "cogr-gr-pos.png", target_dir)
    #     # print("Grasp sin")
    #     draw_lincomb(ids_p, proto_p, sin_coef_p, "cogr-gr-sin.png", target_dir)
    #     # print("Grasp cos")
    #     draw_lincomb(ids_p, proto_p, cos_coef_p, "cogr-gr-cos.png", target_dir)
    #     # print("Grasp wid")
    #     draw_lincomb(ids_p, proto_p, wid_coef_p, "cogr-gr-wid.png", target_dir)

    ins_masks = torch.sigmoid(torch.matmul(protos, ins_coef_pred_kept.t())).contiguous()
    grasp_qua_masks = torch.sigmoid(torch.matmul(protos, grasp_coef_pred_kept[:, 0, :].t())).contiguous()
    grasp_sin_masks = torch.matmul(protos, grasp_coef_pred_kept[:, 1, :].t()).contiguous()
    grasp_cos_masks = torch.matmul(protos, grasp_coef_pred_kept[:, 2, :].t()).contiguous()
    grasp_wid_masks = torch.sigmoid(torch.matmul(protos, grasp_coef_pred_kept[:, 3, :].t())).contiguous()

    ins_masks = crop(ins_masks, box_pred_kept).permute(2,0,1)
    grasp_qua_masks = crop(grasp_qua_masks, box_pred_kept).permute(2,0,1)
    grasp_sin_masks = crop(grasp_sin_masks, box_pred_kept).permute(2,0,1)
    grasp_cos_masks = crop(grasp_cos_masks, box_pred_kept).permute(2,0,1)
    grasp_wid_masks = crop(grasp_wid_masks, box_pred_kept).permute(2,0,1)

    ins_masks = F.interpolate(ins_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    ins_masks.gt_(0.5)
    grasp_qua_masks = F.interpolate(grasp_qua_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    grasp_sin_masks = F.interpolate(grasp_sin_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    grasp_cos_masks = F.interpolate(grasp_cos_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)
    grasp_wid_masks = F.interpolate(grasp_wid_masks.unsqueeze(0), (input_size, input_size), mode='bilinear', align_corners=False).squeeze(0)

    ins_masks = ins_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_qua_masks = grasp_qua_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_sin_masks = grasp_sin_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_cos_masks = grasp_cos_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()
    grasp_wid_masks = grasp_wid_masks[:, 0:ori_h, 0:ori_w].cpu().numpy()

    grasp_ang_masks = []
    for i in range(ins_masks.shape[0]):
        grasp_qua_masks[i] = gaussian(grasp_qua_masks[i], 2.0, preserve_range=True)
        ang_mask = (np.arctan2(grasp_sin_masks[i], grasp_cos_masks[i]) / 2.0)
        grasp_ang_masks.append(ang_mask)
    grasp_ang_masks = np.asarray(grasp_ang_masks)
    scale = np.array([ori_w, ori_w, ori_w, ori_w])
    box_pred_kept = box_pred_kept.cpu().numpy() * scale

    ins_grasp_rects_top1 = []
    ins_grasp_rects_top5 = []
    for i in range(ins_masks.shape[0]):
        grasps_top1, _ = detect_grasps(grasp_qua_masks[i], grasp_sin_masks[i], grasp_cos_masks[i], grasp_wid_masks[i], 1)
        grasps_top5, _ = detect_grasps(grasp_qua_masks[i], grasp_sin_masks[i], grasp_cos_masks[i], grasp_wid_masks[i], 5)
        ins_grasp_rects_top1.append(grasps_top1)
        ins_grasp_rects_top5.append(grasps_top5)
    

    return {
        "cls": class_ids,
        "bboxes": box_pred_kept,
        "ins_masks": ins_masks,
        "grasps_top1": ins_grasp_rects_top1,
        "grasps_top5": ins_grasp_rects_top5,
        "grasp_masks": (grasp_qua_masks, grasp_ang_masks, grasp_wid_masks)
    }




def visualization(img, mask, grasp_masks, grasps, text, save_path=None):
    grasp_qua_mask, grasp_ang_mask, grasp_wid_mask = grasp_masks
    
    fig = plt.figure(figsize=(25, 10))
    
    # draw grasp rectangles in image
    tmp = img.copy()
    for rect in grasps:
        center_x, center_y, width, height, theta = rect
        box = ((center_x, center_y), (width, height), -(theta+180))
        box = cv2.boxPoints(box)
        box = np.intp(box)
        ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in box]
        tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
        tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
        tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
        tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
    # tmp = cv2.rectangle(tmp, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
    tmp = cv2.putText(tmp, text, (0,10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
    
    # draw predicted instance mask in image
    msk_img = (img * 0.3).astype(np.uint8).copy()
    mask = mask.astype(np.uint8)
    msk_img[mask, 0] = 255
    
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(img/255.)
    ax.set_title('RGB')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(tmp/255.)
    ax.set_title('predicted grasps')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(mask)
    ax.set_title('predicted instance mask')
    ax.axis('off')
    
    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_qua_mask, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp quality')
    ax.axis('off')
    plt.colorbar(plot)
    
    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_ang_mask, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Angle')
    ax.axis('off')
    plt.colorbar(plot)
    
    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_wid_mask, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Width')
    ax.axis('off')
    plt.colorbar(plot)
    
    plt.suptitle(f"{text}", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path)
    

def detect_grasps(grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, num_grasps=5):
    grasps = []
    max_width = 100
    local_max = peak_local_max(grasp_quality_mask, min_distance=2, threshold_abs=0.4, num_peaks=num_grasps)
    grasp_angle_mask = (np.arctan2(grasp_sin_mask, grasp_cos_mask) / 2.0)
    
    for p_array in local_max:
        grasp_point = tuple(p_array)
        grasp_angle = grasp_angle_mask[grasp_point] / np.pi * 180
        grasp_width = grasp_wid_mask[grasp_point]

        grasps.append([float(grasp_point[1]), float(grasp_point[0]), grasp_width*max_width, 20, grasp_angle])
    
    return grasps, grasp_angle_mask


def calculate_iou(rect_p, rect_gt, shape=(480, 640), angle_threshold=30):
    if abs(rect_p[4] - rect_gt[4]) > angle_threshold and abs(rect_p[4] + rect_gt[4]) > angle_threshold:
        return 0
    
    center_x, center_y, w_rect, h_rect, theta, _ = rect_gt
    gt_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
    gt_box = cv2.boxPoints(gt_r_rect)
    gt_box = np.int0(gt_box)
    rr1, cc1 = polygon(gt_box[:, 0], gt_box[:,1], shape)

    mask_rr = rr1 < shape[1]
    rr1 = rr1[mask_rr]
    cc1 = cc1[mask_rr]

    mask_cc = cc1 < shape[0]
    cc1 = cc1[mask_cc]
    rr1 = rr1[mask_cc]

    center_x, center_y, w_rect, h_rect, theta = rect_p
    p_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
    p_box = cv2.boxPoints(p_r_rect)
    p_box = np.int0(p_box)
    rr2, cc2 = polygon(p_box[:, 0], p_box[:,1], shape)

    mask_rr = rr2 < shape[1]
    rr2 = rr2[mask_rr]
    cc2 = cc2[mask_rr]

    mask_cc = cc2 < shape[0]
    cc2 = cc2[mask_cc]
    rr2 = rr2[mask_cc]

    area = np.zeros(shape)
    area[cc1, rr1] += 1
    area[cc2, rr2] += 1

    union = np.sum(area > 0)
    intersection = np.sum(area == 2)

    if union <= 0:
        return 0
    else:
        return intersection / union


def calculate_max_iou(rects_p, rects_gt):
    max_iou = 0
    for rect_gt in rects_gt:
        for rect_p in rects_p:
            iou = calculate_iou(rect_p, rect_gt)
            # print("==============================")
            # print(rect_p, rect_gt, iou)
            if iou > max_iou:
                max_iou = iou
    return max_iou


def calculate_jacquard_index(grasp_preds, grasp_targets, iou_threshold=0.25):
    j_index = 0
    grasp_preds = np.asarray(grasp_preds)
    grasp_targets = np.asarray(grasp_targets)

    grasp_targets[:, 3] = 20
    grasp_targets[:, 2] = np.clip(grasp_targets[:, 2], 0, 100)
    
    iou = calculate_max_iou(grasp_preds, grasp_targets)
    if iou > iou_threshold:
        j_index = 1
    
    return j_index
```

## 6.misc.py

```
import os
import random
import numpy as np
from PIL import Image
from loguru import logger
import sys
import inspect

import cv2
import torch
from torch import nn
import torch.distributed as dist


def init_random_seed(seed=None, device='cuda', rank=0, world_size=1):
    """Initialize random seed."""
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.name == "Lr":
            fmtstr = "{name}={val" + self.fmt + "}"
        else:
            fmtstr = "{name}={val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("  ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def trainMetricGPU(output, target, threshold=0.35, pr_iou=0.5, sigmoid=True):
    assert (output.dim() in [2, 3, 4])
    assert output.shape == target.shape
    output = output.flatten(1)
    target = target.flatten(1)
    if sigmoid:
        output = torch.sigmoid(output)
    output[output < threshold] = 0.
    output[output >= threshold] = 1.
    # inter & union
    inter = (output.bool() & target.bool()).sum(dim=1)  # b
    union = (output.bool() | target.bool()).sum(dim=1)  # b
    ious = inter / (union + 1e-6)  # 0 ~ 1
    # iou & pr@5
    iou = ious.mean()
    prec = (ious > pr_iou).float().mean()
    return 100. * iou, 100. * prec


def ValMetricGPU(output, target, threshold=0.35):
    assert output.size(0) == 1
    output = output.flatten(1)
    target = target.flatten(1)
    output = torch.sigmoid(output)
    output[output < threshold] = 0.
    output[output >= threshold] = 1.
    # inter & union
    inter = (output.bool() & target.bool()).sum(dim=1)  # b
    union = (output.bool() | target.bool()).sum(dim=1)  # b
    ious = inter / (union + 1e-6)  # 0 ~ 1
    return ious


def intersectionAndUnionGPU(output, target, K, threshold=0.5):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)

    output = torch.sigmoid(output)
    output[output < threshold] = 0.
    output[output >= threshold] = 1.

    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(),
                                    bins=K,
                                    min=0,
                                    max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection[1], area_union[1]


def group_weight(weight_group, module, lr):
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
    assert len(list(
        module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """
    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")


def get_seg_image(img: np.array, mask: np.array) -> np.array:
    # My stupid way, don't use it...
    # mask = (1 * np.logical_or.reduce(mask)).astype('uint8')
    mask_inv = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    background = cv2.bitwise_and(gray, gray, mask=mask_inv)
    background = np.stack((background,)*3, axis=-1)
    img_ca = res
    # img_ca = cv2.add(res, background)
    return img_ca


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
```

## 7.OCID_sub_class_dict.py

```
import numpy as np

cnames = {
    'background'      : '0',
    'apple'           : '1',
    'ball'            : '2',
    'banana'          : '3',
    'bell_pepper'     : '4',
    'binder'          : '5',
    'bowl'            : '6',
    'cereal_box'      : '7',
    'coffee_mug'      : '8',
    'flashlight'      : '9',
    'food_bag'        : '10',
    'food_box'        : '11',
    'food_can'        : '12',
    'glue_stick'      : '13',
    'hand_towel'      : '14',
    'instant_noodles' : '15',
    'keyboard'        : '16',
    'kleenex'         : '17',
    'lemon'           : '18',
    'lime'            : '19',
    'marker'          : '20',
    'orange'          : '21',
    'peach'           : '22',
    'pear'            : '23',
    'potato'          : '24',
    'shampoo'         : '25',
    'soda_can'        : '26',
    'sponge'          : '27',
    'stapler'         : '28',
    'tomato'          : '29',
    'toothpaste'      : '30',
    'unknown'         : '31'
    }

colors = {
    'b': np.array([0.0, 0.0, 1.0])*255,
    'g': np.array([0.0, 0.5, 0.0])*255,
    'r': np.array([1.0, 0.0, 0.0])*255,
    'c': np.array([0.0, 0.75, 0.75])*255,
    'm': np.array([0.75, 0, 0.75])*255,
    'y': np.array([0.75, 0.75, 0])*255,
     'w': np.array([1.0, 1.0, 1.0])*255,
}

colors_list = list(colors.values())



subnames = {'background': 0,
 'apple_1': 1,
 'apple_2': 2,
 'ball_1': 3,
 'ball_2': 4,
 'ball_3': 5,
 'banana_1': 6,
 'banana_2': 7,
 'bell_pepper_1': 8,
 'binder_1': 9,
 'bowl_1': 10,
 'cereal_box_1': 11,
 'cereal_box_3': 12,
 'cereal_box_4': 13,
 'cereal_box_5': 14,
 'coffee_mug_1': 15,
 'coffee_mug_2': 16,
 'flashlight_1': 17,
 'food_bag_2': 18,
 'food_bag_3': 19,
 'food_bag_4': 20,
 'food_box_1': 21,
 'food_box_2': 22,
 'food_box_3': 23,
 'food_can_1': 24,
 'food_can_2': 25,
 'food_can_3': 26,
 'glue_stick_1': 27,
 'hand_towel_1': 28,
 'hand_towel_2': 29,
 'hand_towel_3': 30,
 'instant_noodles_1': 31,
 'instant_noodles_2': 32,
 'keyboard_1': 33,
 'keyboard_2': 34,
 'kleenex_1': 35,
 'kleenex_2': 36,
 'kleenex_3': 37,
 'lemon_1': 38,
 'lemon_2': 39,
 'lime_1': 40,
 'lime_2': 41,
 'marker_1': 42,
 'marker_2': 43,
 'marker_3': 44,
 'orange_1': 45,
 'orange_2': 46,
 'peach_1': 47,
 'peach_2': 48,
 'pear_1': 49,
 'pear_2': 50,
 'potato_1': 51,
 'potato_2': 52,
 'shampoo_1': 53,
 'shampoo_2': 54,
 'shampoo_3': 55,
 'soda_can_1': 56,
 'soda_can_2': 57,
 'sponge_1': 58,
 'sponge_2': 59,
 'sponge_3': 60,
 'stapler_1': 61,
 'stapler_2': 62,
 'tomato_1': 63,
 'toothpaste_1': 64,
 'toothpaste_2': 65,
 'unknown': 66
}

sub_to_class = {0: 0,
 1: 1,
 2: 1,
 3: 2,
 4: 2,
 5: 2,
 6: 3,
 7: 3,
 8: 4,
 9: 5,
 10: 6,
 11: 7,
 12: 7,
 13: 7,
 14: 7,
 15: 8,
 16: 8,
 17: 9,
 18: 10,
 19: 10,
 20: 10,
 21: 11,
 22: 11,
 23: 11,
 24: 12,
 25: 12,
 26: 12,
 27: 13,
 28: 14,
 29: 14,
 30: 14,
 31: 15,
 32: 15,
 33: 16,
 34: 16,
 35: 17,
 36: 17,
 37: 17,
 38: 18,
 39: 18,
 40: 19,
 41: 19,
 42: 20,
 43: 20,
 44: 20,
 45: 21,
 46: 21,
 47: 22,
 48: 22,
 49: 23,
 50: 23,
 51: 24,
 52: 24,
 53: 25,
 54: 25,
 55: 25,
 56: 26,
 57: 26,
 58: 27,
 59: 27,
 60: 27,
 61: 28,
 62: 28,
 63: 29,
 64: 30,
 65: 30,
 66: 31
}

```

## 8.simple_tokenizer.py

**文本分词器**。用于处理输入文本，将其转换为 CLIP 模型所需的 Token 序列（BPE 编码）

```
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

```

