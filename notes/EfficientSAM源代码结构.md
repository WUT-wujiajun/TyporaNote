# EfficientSAM源代码结构

## efficient_sam:

### 1.build_efficient_sam.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 从efficient_sam.py中导入build_efficient_sam函数
from .efficient_sam import build_efficient_sam

# 返回ViT-Tiny版本的efficientsam模型
def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="weights/efficient_sam_vitt.pt",
    ).eval()

# 返回ViT-Small版本的efficientsam模型
def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint="weights/efficient_sam_vits.pt",
    ).eval()

```

### 2.efficient_sam_decoder.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPBlock

# 提示词编码器，将用户给出的几何提示（点、框）翻译成 Transformer 能够理解的高维特征向量
# 融合位置信息，以及点击位置所含的标签信息
class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
    ) -> None:
        """
        对提示信息进行编码，以供 SAM 的掩码解码器输入使用

        Arguments:
          embed_dim (int): 提示信息的嵌入维度
          image_embedding_size (tuple(int, int)): 图像嵌入的空间尺寸，格式为(H，W)
          input_image_size (int): 输入到图像编码器的图像经过填充后的尺寸，格式为(H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)# 因为_pe_encoding 最后会把 sin 和 cos 拼起来
        # Label = -1: 无效点/填充点 (Padding)
        self.invalid_points = nn.Embedding(1, embed_dim)
        # Label = 1: 正向点 (用户点的那个位置，表示"我要这个")
        self.point_embeddings = nn.Embedding(1, embed_dim)
        # Label = 2: 框的左上角 (Box Top-Left)
        self.bbox_top_left_embeddings = nn.Embedding(1, embed_dim)
        # Label = 3: 框的右下角 (Box Bottom-Right)
        self.bbox_bottom_right_embeddings = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        返回用于编码点提示信息的位置编码，该编码会应用于一组密集的点集合，这些点的形状与图像编码的形状一致.

        Returns:
          torch.Tensor: 位置编码，形状为： 1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        # 这里的 self.image_embedding_size 通常是 (64, 64)
        # self.pe_layer 会生成一个 [C, 64, 64] 的位置编码张量
        # .unsqueeze(0) 是为了增加 Batch 维度，变成 [1, C, 64, 64]
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,# 形状 [B, N, 2]，坐标值
        labels: torch.Tensor,# 形状 [B, N]，整数标签（1, 2, 3, -1）
    ) -> torch.Tensor:
        """编码位置提示."""
        points = points + 0.5  # 像素中心化偏移
        # 用户输入点转为编码
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        # 制作掩码 (Masks)
        invalid_label_ids = torch.eq(labels, -1)[:,:,None]# 形状：[Batch, N]->[Batch, N, 1]
        point_label_ids = torch.eq(labels, 1)[:,:,None]
        topleft_label_ids = torch.eq(labels, 2)[:,:,None]
        bottomright_label_ids = torch.eq(labels, 3)[:,:,None]
        # 如果这个点是"无效点" (-1)，就加上 invalid_points 的权重，[1,1,embed_dim]*[Batch, N, 1]=[Batch, N, embed_dim]
        point_embedding = point_embedding + self.invalid_points.weight[:,None,:] * invalid_label_ids
        # 如果这个点是"正向点" (1)，就加上 point_embeddings 的权重
        point_embedding = point_embedding + self.point_embeddings.weight[:,None,:] * point_label_ids
        point_embedding = point_embedding + self.bbox_top_left_embeddings.weight[:,None,:] * topleft_label_ids
        point_embedding = point_embedding + self.bbox_bottom_right_embeddings.weight[:,None,:] * bottomright_label_ids
        return point_embedding # [Batch, N, embed_dim]

    def forward(
        self,
        coords,
        labels,
    ) -> torch.Tensor:
        """
        对不同类型的提示信息进行嵌入，返回稀疏嵌入和密集嵌入两种结果.

        Arguments:
          points: 形状为 [B，N,2] 的张量
          labels: 形状为 [B,N] 的整数张量，其中每个元素的值为 1、2 或 3.

        Returns:
          torch.Tensor: 点和框对应的稀疏嵌入，形状为：BxNx(embed_dim), 其中N由输入点和框的数量决定

        注意：在完整的 SAM 代码中，Box 通常会先被拆解成两个点（左上角点和右下角点），
            然后分别标记 Label 为 2 和 3，最后像处理点一样传入这个函数。
            所以这里只需要处理 coords 和 labels 即可兼容点和框
        """
        return self._embed_points(coords, labels)

# 位置编码，生成图本身的位置编码以及用户点击位置的编码
class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率的位置编码
    与常见的 Transformer 中的正弦位置编码（Sinusoidal PE）或可学习位置编码（Learnable PE）不同
    """

    def __init__(self, num_pos_feats: int) -> None:
        super().__init__()
        # 初始化了一个形状为 [2, N] 的矩阵，2代表（x,y），num_pos_feats是位置编码的特征维度
        self.register_buffer(
            "positional_encoding_gaussian_matrix", torch.randn((2, num_pos_feats))
        )

    # 把低维坐标(x, y)映射到高维向量
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """对归一化到 [0, 1] 区间的点进行位置编码."""
        # [Step 1] 坐标映射到 [-1, 1]
        # 输入 coords 范围是 [0, 1]
        # 2 * [0, 1] - 1 = [-1, 1]，这样做是为了让坐标以 0 为中心对称。
        coords = 2 * coords - 1
        # [Step 2] 线性投影 (矩阵乘法)
        # coords 形状: [..., 2] (最后的维度是 x,y)
        # matrix 形状: [2, num_pos_feats]
        # 结果形状: [..., num_pos_feats]
        # 这一步把 2D 坐标随机投影到了 N 维空间，[H, W, 2]->[H, W, num_pos_feats]
        coords = coords @ self.positional_encoding_gaussian_matrix
        # [Step 3] 乘以 2π
        # 为放入 sin/cos 函数做准备
        coords = 2 * np.pi * coords
        # [Step 4] 傅里叶特征变换 (sin, cos)
        # 输入形状: [..., num_pos_feats]
        # torch.cat 拼接后形状: [..., 2*num_pos_feats]
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    # 给图像特征图（Image Embedding）生成位置编码
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """为指定尺寸的网格生成位置编码."""
        h, w = size # 比如输入 (64, 64)
        device = self.positional_encoding_gaussian_matrix.device
        # [Step 1] 创建基础网格
        # 生成一个全是 1 的矩阵
        grid = torch.ones([h, w], device=device, dtype=torch.float32)
        # [Step 2] 计算像素坐标 (cumsum)
        # cumsum(dim=0): 沿高度累加 -> [1, 2, 3, ..., h]
        # cumsum(dim=1): 沿宽度累加 -> [1, 2, 3, ..., w]
        # - 0.5 的作用是【像素中心化 (Pixel Centering)】
        # 例如：第 1 个像素原本是 1，减去 0.5 变成 0.5（像素中心），处理之后y_embed形如
        # =0.5 & 0.5 & 0.5 
        #  1.5 & 1.5 & 1.5 
        #  2.5 & 2.5 & 2.5
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        # [Step 3] 归一化到 [0, 1]
        # 除以高度 h 和宽度 w。
        # 这样无论图像多大，坐标范围都在 0 到 1 之间。
        # 这是 SAM 能够处理任意分辨率的关键。
        y_embed = y_embed / h
        x_embed = x_embed / w

        # [Step 4] 拼接x和y编码，stack 后形状: [h, w, 2]，_pe_encoding后形状[h, w, 2*num_pos_feats]
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # [Step 5] 调整维度顺序
        # permute 后: [C, h, w] 
        return pe.permute(2, 0, 1)  # C x H x W
    
    # 用于给用户的点击（Prompt）生成位置编码
    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """对未归一化到 [0, 1] 区间的点进行位置编码."""
        # coords_input 形状通常是 [Batch, Num_Points, 2]
        # 里面的值是绝对像素坐标，比如 (512, 300)

        coords = coords_input.clone()
        # [Step 1] 归一化坐标
        # 把绝对像素坐标除以图像总长宽，变成 [0, 1] 之间的小数。
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

# 掩码解码器
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        upscaling_layer_dims: List[int],
    ) -> None:
        """
        基于图像和提示嵌入，使用 Transformer 架构预测掩码.

        Arguments:
          transformer_dim (int): Transformer 的通道维度
          transformer (nn.Module): 用于预测掩码的 Transformer 模块
          num_multimask_outputs (int): 掩码消歧时需要预测的掩码数量
          activation (nn.Module): 掩码上采样时使用的激活函数类型
          iou_head_depth (int): 用于预测掩码质量的 MLP 深度
          iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # 1. Output Tokens (输出令牌)
        # 类似于 BERT 中的 [CLS] token。
        # iou_token: 专门用来负责预测 "这个掩码质量好不好" (IoU Score)
        self.iou_token = nn.Embedding(1, transformer_dim)
        if num_multimask_outputs > 1:
            self.num_mask_tokens = num_multimask_outputs + 1 # 1个基础+3个消歧义掩码
        else:
            self.num_mask_tokens = 1
        
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        output_dim_after_upscaling = transformer_dim
        # 构建上采样层final_output_upscaling_layers
        self.final_output_upscaling_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.final_output_upscaling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    activation(),
                )
            )
            output_dim_after_upscaling = layer_dims
        # 构建output_hypernetworks_mlps
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                # 使用列表推导式，为每一个 Mask Token 创建一个独立的 MLP
                # self.num_mask_tokens 通常是 4 (对应 3 个不同层级的掩码 + 1 个可能的额外掩码)
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=output_dim_after_upscaling,
                    num_layers=2,
                    act=activation,
                )
                for i in range(self.num_mask_tokens)
            ]
        )
        # 构建 IoU 预测头
        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,# 比如生成了 3 个掩码，就输出 3 个分数 (例如 [0.95, 0.4, 0.2])
            num_layers=iou_head_depth,
            act=activation,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于图像嵌入和提示嵌入预测掩码.

        Arguments:
          image_embeddings: 形状为[B, C, H, W] or [B*max_num_queries, C, H, W]的张量
          image_pe (torch.Tensor): 与图像嵌入形状一致的位置编码（批次维度支持广播）
          sparse_prompt_embeddings (torch.Tensor): 点和框对应的稀疏提示嵌入
          multimask_output (bool): 是否返回多个掩码，而非单个掩码

        Returns:
          torch.Tensor: 批量的预测掩码
          torch.Tensor: 批量的掩码质量预测结果
        """

        (
            batch_size,
            max_num_queries,
            sparse_embed_dim_1,
            sparse_embed_dim_2,
        ) = sparse_prompt_embeddings.shape

        (
            _,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        ) = image_embeddings.shape

        # 假设image_embeddings 形状：[2, 256, 64, 64]（两张图），max_num_queries = 3 (每张图3个提示)
        # 图像嵌入处理
        image_embeddings_tiled = torch.tile(
            image_embeddings[:, None, :, :, :], [1, max_num_queries, 1, 1, 1]# [2, 256, 64, 64]->[2, 1, 256, 64, 64]->[2, 3, 256, 64, 64]
        ).view(
            batch_size * max_num_queries,# [2, 3, 256, 64, 64]->[6, 256, 64, 64]
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        )
        # 提示嵌入处理
        sparse_prompt_embeddings = sparse_prompt_embeddings.reshape(
            batch_size * max_num_queries, sparse_embed_dim_1, sparse_embed_dim_2
        )
        # 预测掩码与iou
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings_tiled,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        # 如果开启多掩码输出，取Index 1, 2, 3，丢弃 Index 0 结果，反之，只取Index 0 结果
        if multimask_output and self.num_multimask_outputs > 1:
            return masks[:, 1:, :], iou_pred[:, 1:]
        else:
            return masks[:, :1, :], iou_pred[:, :1]
        
    # 预测掩码函数
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,# 图像编码器输出的图像特征
        image_pe: torch.Tensor,# 图像的位置编码
        sparse_prompt_embeddings: torch.Tensor,# 用户输入的点 / 框等提示的编码结果
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        sparse_prompt_embeddings (用户提示) —— 是“题目”
        output_tokens (输出令牌) —— 是“空白答题卡”
        把它们拼接到同一个序列 [Batch, 5+N, 256] 中，
        是为了让它们在 Transformer 内部发生 自注意力 (Self-Attention) 交互
        '''
        # 1. 拼接权重
        # self.iou_token.weight: [1, 256] -> 负责预测质量的令牌
        # self.mask_tokens.weight: [4, 256] -> 负责生成掩码的令牌
        # cat 之后: [5, 256]
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        # 2. 扩展维度 (Expand)
        # 结果形状: [5, 256]->[1, 5, 256]->[Batch, 5, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        # 3. 全员集合 (Concatenate)
        # output_tokens: [Batch, 5, 256] (输出占位符)
        # sparse_prompt_embeddings: [Batch, N, 256] (用户的点击/框提示)
        # 这就是送入 Transformer 的 Query 序列。
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # 拼接后 tokens: [Batch, 5+N, 256]


        # 1. 复制位置编码
        # image_pe 原本是 [1, 256, 64, 64] (所有图共享同一套位置编码)。
        # 但 tokens 有 Batch 维度，所以要把 PE 也复制成同样的 Batch 大小->[B, 256, 64, 64]
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # 2. 获取图像特征的形状 (为了后面 reshape 用)
        b, c, h, w = image_embeddings.shape

        # 调用 Two-Way Transformer
        # image_embeddings: 图像特征 (Key/Value)
        # pos_src: 图像位置编码
        # tokens: 所有的令牌 (Query)

        # 这个函数内部发生了什么？
        # 1. Self-Attention: Tokens 之间交流 (比如 IoU Token 问 Mask Token: "你打算画哪？")
        # 2. Cross-Attention (Token to Image): Tokens 去看图像 ("这个坐标对应的是红色的像素吗？")
        # 3. Cross-Attention (Image to Token): 图像反过来关注 Tokens ("这块红色像素是被点中的吗？")

        # 返回值解析：
        # hs: 更新后的 Tokens [Batch, 5+N, 256]。它们现在“肚子里有货了”。
        # src: 更新后的图像特征 [Batch, 4096, 256]。它现在的特征更聚焦于提示点附近的物体。
        hs, src = self.transformer(image_embeddings, pos_src, tokens)

        # 1. 提取 IoU Token
        # 取出第 0 个 token，它包含了"掩码质量"的信息。
        iou_token_out = hs[:, 0, :] # [B, 256]
        # 2. 提取 Mask Tokens
        # 取出第 1 到第 5 个 token (共4个)
        # 它们包含了"如何生成掩码"的指令信息。
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]# [B, 4, 256]

        # 1. 恢复 2D 形状
        # src 原本是序列格式 [Batch, 4096, 256]
        # transpose + view -> [Batch, 256, 64, 64]
        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)

        # 2. 上采样 (Upscaling)
        # 通过之前定义的 ConvTranspose2d 层循环处理。
        # 比如经过两层：64x64 -> 128x128 -> 256x256。
        # 通道数也会减少 (比如从 256 降到 32)，让计算更轻量。
        for upscaling_layer in self.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)# [Batch, 32, 256, 256]
        # 这就是我们最终用来生成掩码的"精细画布"。
        hyper_in_list: List[torch.Tensor] = []

        # 遍历每一个 Mask Token (比如有 4 个)
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            # output_hypernetworks_mlp 是一个小 MLP。
            # 输入: Mask Token [Batch, 256]
            # 输出: 动态权重 [Batch, 32]
            # 结果存入hyper_in_list
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        
        # 把list里的四个结果堆叠起来
        # hyper_in 形状: [Batch, 4, 32]
        # 含义: 这里的 4 代表我们要同时预测 4 张掩码。32 代表每个掩码对应的"画笔参数"。
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 获取画布尺寸
        b, c, h, w = upscaled_embedding.shape

        # 核心运算公式: Masks = Weights * Features
        # 1. upscaled_embedding.view(...) -> 把画布拉平: [Batch, 32, H*W]
        # 2. hyper_in: [Batch, 4, 32]
        # 3. @ (矩阵乘法): [Batch, 4, 32] x [Batch, 32, H*W] -> [Batch, 4, H*W]
        # 4. .view(...) -> 恢复 2D: [Batch, 4, H, W]

        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # 这里的 masks 里面的数值是 logits (未归一化的分数)。
        # 大于 0 的地方通常被认为是前景。

        # 使用 IoU Head 对 iou_token_out 进行预测
        # 输入: [Batch, 256]
        # 输出: [Batch, 4] (对应 4 张掩码的分数)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

```

### 3.efficient_sam_encoder.py

```py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# 专门针对 2D 图像格式 [B, C, H, W] 进行层归一化
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# [B, 3, 1024, 1024]->[B, embed_dim, 64, 64]
# [B, 3, 1024, 1024]->[B, 384, 64, 64]
class PatchEmbed(nn.Module):
    """把图像切块并变成向量"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,# 输入通道数
        embed_dim,# 输出通道数，也指向量维度
    ):
        super().__init__()
        # 卷积核大小和步长相同
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x

# 多头注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,# 总维数
        num_heads,# 头数
        qkv_bias,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 计算每个头的维数
        self.scale = qk_scale or head_dim**-0.5 # 缩放因子，即 1/sqrt(d_k)
        # 关键点：用一个大的全连接层同时生成 Q, K, V
        # 输入 dim，输出 3倍 dim (分别对应 Q, K, V)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x) # 输入 [B, N, C] -> 输出 [B, N, 3*C]
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) # 形状变为：[B, N, 3, num_heads, head_dim]
            .permute(2, 0, 3, 1, 4) # 形状变为：[3, B, num_heads, N, head_dim]
        )
        # 现在 q, k, v 的形状都是 [B, num_heads, N, head_dim]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        # 矩阵乘法 Q @ K.T，形状变化：[..., N, head_dim] @ [..., head_dim, N] -> [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 形状：[..., N, N] @ [..., N, head_dim] -> [B, num_heads, N, head_dim]-> [B, N, num_heads, head_dim]->[B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 最后全连接层，混合不同头的信息
        x = self.proj(x)
        return x

# 多层感知机，线性层和激活函数层构成
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,# 输入维度，比如768
        hidden_features=None,# 隐藏层维度 (通常是输入的 4 倍，例如 3072)
        out_features=None,# 输出维度 (通常回到 768)
        act_layer=nn.GELU,# 激活函数 (默认 GELU)
    ):
        super().__init__()
        # 如果没指定输出/隐藏维度，默认和输入一样
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一层：升维 (Expansion)
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数：增加非线性
        self.act = act_layer()
        # 第二层：降维 (Reduction)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# 构建一个由（归一化+自注意力机制+归一化+MLP）+残差连接的block
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        # 1. 第一个归一化层 (用于 Attention 之前)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # 2. 多头自注意力机制 (负责"看全局"，提取上下文信息)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        # 3. 第二个归一化层 (用于 MLP 之前)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # 4. 多层感知机 (负责"深思考"，提炼特征)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        # Attention + 残差
        x = x + self.attn(self.norm1(x))
        # MLP + 残差
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    计算绝对位置嵌入。若需要，调整嵌入的大小并针对原始嵌入移除分类标记（cls_token）维度。
    Args:
        abs_pos (Tensor): 绝对位置嵌入，形状为 (1, num_position, C)。
        has_cls_token (bool): 若为 True，abs_pos 中包含 1 个用于分类标记的嵌入。
        hw (Tuple): 输入图像标记的尺寸。

    Returns:
        处理后的绝对位置嵌入，形状为 (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    # 如果有CLS Token，只取第 1 位之后的数据
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    # 取出当前num_position并计算尺寸size
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num
    # 把尺寸缩放到h和w，返回形状（1，h，w，c）
    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


# efficient SAM的图像编码器
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int,# 输入图像尺寸
        patch_size: int,# Patch 尺寸
        in_chans: int,# 输入图像通道数
        patch_embed_dim: int,# Patch 嵌入维度
        normalization_type: str,
        depth: int,#  ViT 的深度（Block 数量）
        num_heads: int,# 每个 ViT 块中的注意力头数
        mlp_ratio: float,# MLP 隐藏层维度与嵌入维度的比值
        neck_dims: List[int],
        act_layer: Type[nn.Module],# 激活函数层
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.img_size = img_size
        # 计算被分成多少格子长度，如1024 // 16 = 64
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        # 224 是指【预训练模型】的原始分辨率（通常是 ImageNet 预训练的 ViT）
        pretrain_img_size = 224
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        # 1. 计算预训练时的 Patch 总数。
        # (224 // 16) * (224 // 16) = 14 * 14 = 196 个 Patch
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        # 2. 计算总位置数。
        # 为什么要 +1？是为了给 [CLS] Token (分类 Token) 留一个位置。
        # 虽然 SAM 分割任务可能不用 CLS Token，但为了加载预训练权重，结构必须对齐
        num_positions = num_patches + 1
        # 3. 创建可学习的参数 pos_embed
        # 形状是 [1, 197, patch_embed_dim]。初始化为全 0，会在加载权重时被覆盖。
        # 注意：这里创建的是 14x14 的位置编码，后面 forward 里的 get_abs_pos 会负责把它拉伸成 64x64
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        # 创建模型列表
        self.blocks = nn.ModuleList()
        # 循环 depth 次 (例如 12 次)
        for i in range(depth):
            # 实例化一个前面定义的Block (包含 Attention 和 MLP)
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True)
            # 把它加到模型列表中
            self.blocks.append(vit_block)
        self.neck = nn.Sequential(
            # 1. 1x1 卷积：调整通道数 (降维)
            # 例如从 384 降到 256 (neck_dims[0])
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            # 2. 归一化：也就是最开始定义的 LayerNorm2d
            LayerNorm2d(neck_dims[0]),
            # 3. 3x3 卷积：空间平滑
            # 进一步融合局部特征，消除棋盘格效应。
            # padding=1 保证长宽不变
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            # 4. 再次归一化
            LayerNorm2d(neck_dims[0]),
        )

    '''
    假设输入一张 1024x1024 的 RGB 图片，Patch Size 为 16，patch_embed_dim为 384

    总结数据流：
        图片进：[B, 3, 1024, 1024]
        切块：[B, 384, 64, 64]
        变序列：[B, 4096, 384] (加位置编码，过 Transformer)
        变回图：[B, 384, 64, 64]
        Neck调整：[B, 256, 64, 64]
        特征出。
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入的图片长宽是否等于模型初始化时设定的 img_size (例如 1024)
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        # [B, 3, 1024, 1024]->[B, 384, 64, 64]
        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        # 注入位置信息，调用 get_abs_pos，把预训练的14x14位置编码，插值拉伸成64x64，并把编码加到x上
        # 输出：[B, 64, 64, 384] (形状不变，数值变了)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )
        # 获取当前边长 64
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches # 确保是正方形
        # 展平操作，[B, 64, 64, 384]->[B, 4096, 384]
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        # 数据穿过12层Transformer Block
        for blk in self.blocks:
            x = blk(x)
        #[B, 4096, 384]->[B, 64, 64, 384]
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        # 1. permute: [B, 64, 64, 384] -> [B, 384, 64, 64] (变回 CNN 格式)
        # 2. self.neck: 卷积降维和平滑
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

```

### 4.efficient_sam.py

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from .efficient_sam_decoder import MaskDecoder, PromptEncoder
from .efficient_sam_encoder import ImageEncoderViT
from .two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer

# 模型的主体部分
class EfficientSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    # ：后面跟的是变量类型，属于类型注释
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM 可根据图像和输入提示预测目标掩码.

        Arguments:
          image_encoder (ImageEncoderViT): 用于将图像编码为图像嵌入的骨干网络，这些嵌入可实现高效的掩码预测.
          prompt_encoder (PromptEncoder): 对各类输入提示进行编码.
          mask_decoder (MaskDecoder): 根据图像嵌入和编码后的提示预测掩码.
          pixel_mean (list(float)): 用于归一化输入图像像素的均值.
          pixel_std (list(float)): 用于归一化输入图像像素的标准差.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder

        # 存储模型固定参数，转成torch张量形式
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

    # 将方法导出为模型公共接口
    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据image embeddings和prompts预测掩码。此方法仅运行解码器。

        Arguments 参数:
          image_embeddings: 形状为 [B, C, H, W] 或 [B*max_num_queries, C, H, W] 的张量
          batched_points: 形状为 [B, max_num_queries, num_pts, 2] 的张量         
          batched_point_labels: 形状为 [B, max_num_queries, num_pts] 的张量
        Returns 返回值:
          两个张量组成的元组：
            low_res_mask: 形状为 [B, max_num_queries, 256, 256] 的预测掩码张量
            iou_predictions: 形状为 [B, max_num_queries] 的 IOU 分数估计张量
        """

        # 批量大小，最大查询数量，为了分割一个query点击的数量，还有一个_指的是2，也就是（x,y）两个坐标值
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]# 这一条好像重复了
        # 将“原始图像坐标系”下的点坐标，映射（缩放）到“模型输入图像坐标系”中
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        # 无论用户实际输入了多少个num_pts，都要将数量调整为self.decoder_max_num_input_points
        # 如果数量超过解码器支持的最大值
        if num_pts > self.decoder_max_num_input_points:
            # 则保留第0，1，3维度的所有数据，对第2维度，即num_pts所在维度切片，保留前self.decoder_max_num_input_points个点
            rescaled_batched_points = rescaled_batched_points[
                :, :, : self.decoder_max_num_input_points, :
            ]
            # 对应的标签也去除
            batched_point_labels = batched_point_labels[
                :, :, : self.decoder_max_num_input_points
            ]
        #如果数量小于解码器支持的最大值
        elif num_pts < self.decoder_max_num_input_points:
            # 用-1.0进行填充不足的部分
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        # 提示词编码器计算得到稀疏嵌入
        sparse_embeddings = self.prompt_encoder(
            # 合并Batch 和 Queries 两个维度，将合并后的points和labels传入prompt_encoder中
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        # 将稀疏嵌入维度从 [B*Q, N, C] 恢复成 [B, Q, N, C]
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )
        # 掩码解码器计算得到低分辨率掩码low_res_masks，形状[Batch*Queries, num_predictions, 256, 256]和iou预测分数，形状[Batch*Queries, num_predictions]
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        # num_predictions: 每个 Query 生成了几个掩码（通常是 3 个，代表“整体”、“部分”等不同层级）
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        # 结果后处理：将低分辨率掩码缩放至(output_h, output_w)尺寸，一般是原图尺寸
        # 如果要求缩放
        if output_w > 0 and output_h > 0:
            # 将最后两个维度，H和W缩放到output_h和output_w
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            # 输出掩码4维转5维
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        # 不要求缩放
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )
        # iou预测从2维转为3维
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )
        # 返回5维的输出掩码以及3维的iou预测分数
        return output_masks, iou_predictions
    
    #将坐标从原图尺寸安全地转换到模型尺寸（通常是 1024x1024）
    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    # 条件 x坐标>= 0
                    batched_points[..., 0] >= 0,
                    # 如果为真，Xnew=Xold*(1024/原图宽度)，括号内其实就是坐标缩放比例 
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    # 如果为假
                    -1.0,
                ),
                # 同理，处理y坐标
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    @torch.jit.export
    # 把预处理后的图像输入图像编码器，并返回处理结果
    def get_image_embeddings(self, batched_images) -> torch.Tensor:
        """
        端到端地根据提供的图像和提示预测掩码.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: 形状为 [B, 3, H, W] 的张量
        Returns:
          图像嵌入列表，每个嵌入的形状为 [B, C(i), H(i), W(i)].
          最后一个嵌入对应最终层.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
          batched_images: 形状为 [B, 3, H, W] 的张量
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          由两个张量组成的元组列表，其中第 i 个元素是考虑前 i+1 个点得到的结果.
            low_res_mask: 形状为 [B, 256, 256] 的预测掩码张量
            iou_predictions: 形状为 [B, max_num_queries] 的 IOU 分数估计张量
        """
        batch_size, _, input_h, input_w = batched_images.shape
        # 经图像编码器得到图像嵌入
        image_embeddings = self.get_image_embeddings(batched_images)
        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )

    # 图像预处理
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """归一化像素值并填充为正方形输入."""
        # 如果输入图片的H或者W不等于图像编码器要求的尺寸，就将图片强制拉伸/缩放到要求的大小，通常是(1024, 1024)
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam(encoder_patch_embed_dim, encoder_num_heads, checkpoint=None):
    img_size = 1024 # 输入图像必须是 1024x1024
    encoder_patch_size = 16 # ViT 把图像切成 16x16 的块
    encoder_depth = 12 # 编码器有 12 层 Transformer
    encoder_mlp_ratio = 4.0 # MLP膨胀系数，隐藏层的宽度是输入宽度的多少倍
    encoder_neck_dims = [256, 256] # EfficientSAM 特有的优化，颈部维度
    decoder_max_num_input_points = 6 # 模型一次推理最多能处理多少个用户点击的点
    decoder_transformer_depth = 2 # 解码器层数
    decoder_transformer_mlp_dim = 2048 # 解码器 MLP 宽度
    decoder_num_heads = 8 # 解码器的注意力头数
    decoder_upscaling_layer_dims = [64, 32] # 解码器上采样层维度
    num_multimask_outputs = 3 #多义性输出数量，面对一个点，模型会同时预测 3 个可能的掩码（通常代表：整体、部分、子部分）
    iou_head_depth = 3 # IOU预测网络的MLP层数
    iou_head_hidden_dim = 256 # IoU 预测网络的宽度
    activation = "gelu" # 激活函数类型
    normalization_type = "layer_norm" # 归一化方式，使用层归一化
    normalize_before_activation = False # 归一化与激活函数顺序

    # 断言语句，强制检查activation值必须是"relu"或"gelu"，否则抛出错误
    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    # 定义图像编码器，内部结构在efficient_sam_encoder.py实现
    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        # 传入定义好的图像编码器
        image_encoder=image_encoder,
        # 定义并传入提示词编码器
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        # 传入定义好的解码器支持处理最大点数
        decoder_max_num_input_points=decoder_max_num_input_points,
        # 定义并传入掩码解码器
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        # 传入均值和方差
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"])
    return sam

```

### 5.mlp.py

```py
from typing import Type

from torch import nn


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,# 输入向量的维度
        hidden_dim: int,# 中间隐藏层的维度
        output_dim: int,# 最终输出的维度
        num_layers: int,# 隐藏层的数量 (Linear + Activation 的组数)
        act: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        # 1. 准备隐藏层的输入维度列表
        # 如果 num_layers=3，hidden_dim=256
        # 那么 h = [256, 256] (长度为 num_layers - 1)
        h = [hidden_dim] * (num_layers - 1)
        # 2. 构建层列表 (nn.ModuleList)
        self.layers = nn.ModuleList(
            # 这是一个生成器表达式
            nn.Sequential(nn.Linear(n, k), act())
            # zip 把两个列表配对：
            # 左边列表 (输入维度): [input_dim] + h  -> [input, hidden, hidden...]
            # 右边列表 (输出维度): [hidden_dim] * num_layers -> [hidden, hidden, hidden...]
            for n, k in zip([input_dim] + h, [hidden_dim] * num_layers)
        )
        # 3. 最后一层线性层 (没有激活函数)
        # 专门用于映射到最终所需的 output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 1. 依次通过所有隐藏层 (Linear -> Activation)，[B, N, input_dim] ->[B, N, hidden_dim]
        for layer in self.layers:
            x = layer(x)
        # 2. 通过最后一层 (仅 Linear，无激活),[B, N, hidden_dim] ->[B, N, output_dim]
        return self.fc(x)

```

### 6.two_way_transformer.py

```
import math
from typing import Tuple, Type
import torch
from torch import nn, Tensor
from .mlp import MLPBlock




class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module],
        normalize_before_activation: bool,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            curr_layer = TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                normalize_before_activation=normalize_before_activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            self.layers.append(curr_layer)

        self.final_attn_token_to_image = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for idx, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module],
        normalize_before_activation: bool,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = AttentionForTwoWayAttentionBlock(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(
            embedding_dim,
            mlp_dim,
            embedding_dim,
            1,
            activation,
        )

        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if not self.skip_first_layer_pe:
            queries = queries + query_pe
        attn_out = self.self_attn(q=queries, k=queries, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class AttentionForTwoWayAttentionBlock(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."
        self.c_per_head = self.internal_dim / num_heads
        self.inv_sqrt_c_per_head = 1.0 / math.sqrt(self.c_per_head)

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # The fan_out is incorrect, but matches pytorch's initialization
        # for which qkv is a single 3*embedding_dim x embedding_dim matrix
        fan_in = self.embedding_dim
        fan_out = 3 * self.internal_dim
        # Xavier uniform with our custom fan_out
        bnd = math.sqrt(6 / (fan_in + fan_out))
        nn.init.uniform_(self.q_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.k_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.v_proj.weight, -bnd, bnd)
        # out_proj.weight is left with default initialization, like pytorch attention
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn * self.inv_sqrt_c_per_head
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

```

