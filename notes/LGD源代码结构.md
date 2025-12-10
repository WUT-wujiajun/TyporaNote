# LGD源代码结构

数据集结构：

Grasp-Anything/  # 对应 --dataset-path 指定的路径 

├── image/                # 存放 RGB 图像 

│   └── *.jpg             # 图像文件（如：xxx.jpg） 

├── mask/                 # 存放掩码文件

 │   └── *.npy             # 掩码文件（与图像对应，如：xxx.npy） 

├── scene_description/               # 存放提示文本数据 

│   └── *.pkl             # 提示文件（如：xxx.pkl） 

├── grasp_label_positive/                # 存放整体级正样本抓取数据

│   └── *.pt         # 抓取标注文件（如：xxx_0.pt） 

└── grasp-anything++/     # 对应 --add-file-path 指定的路径（示例中为 data/grasp-anything++/seen）    

​	└── seen/        

​		├── grasp_label_positive/  # 存放部位级正样本抓取数据       

​		│      └── *.pt         # 抓取标注文件（如：xxx_0_0.pt）        

​		└── grasp_instructions/  # 存放抓取指令            

​			└── *.pkl            # 指令文件

举例：

**image文件名：**

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830.jpg

（包含一只鸭子和一个苹果的图片）

**mask文件名：**

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_1.npy

（鸭子的整体掩码）

**scene_description文件名：**

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830.pkl

（ 'A small green apple and a yellow rubber duck sitting on a wooden table',
    ['apple', 'duck']）

**grasp_label_positive文件名：**

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0.pt

（苹果的整体抓取标注）

**grasp_instructions文件名：**

苹果有四个部位的抓取描述，第一个0代表苹果，后面的0、1、2、3代表4种抓取方式

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_0.pkl

包含信息：'Pick up apple by its skin.'

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_1.pkl

包含信息：'Take apple by its flesh.'

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_2.pkl

包含信息：'Pick up apple by its seeds.'

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_3.pkl

包含信息：'Take hold of apple on its stem.'

**part_mask文件名：**

苹果4个部位的掩码

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_0.npy

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_1.npy

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_2.npy

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_3.npy

**grasp_label_positive文件名：**

苹果四个部位的抓取标注

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_0.pt

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_1.pt

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_2.pt

000018668931a8fb14891fa2b4c0aaa4b50334d17c20d7d2f6306cc47b2f9830_0_3.pt

# 1.diffusion

## dist_util.py

```
"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

```

## fp16_util.py

```
"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        for p in self.master_params:
            p.grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)

```

## gaussian_diffusion.py

```
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import torch.nn.functional as F

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class NCELoss(th.nn.Module):
    def __init__(self):
        super(NCELoss, self).__init__()

    def forward(self, x_t, guiding_point, model_output):
        # Ensure that all inputs have the same shape
        assert x_t.shape == guiding_point.shape == model_output.shape, "All inputs must have the same shape"
        
        # Flatten the tensors to 2D (batch_size, -1) to compute pairwise distances
        x_t_flat = x_t.view(x_t.size(0), -1)
        guiding_point_flat = guiding_point.view(guiding_point.size(0), -1)
        model_output_flat = model_output.view(model_output.size(0), -1)
        
        # Calculate Euclidean distance for positive samples (model_output and guiding_point)
        positive_distance = F.pairwise_distance(model_output_flat, guiding_point_flat)
        
        # Calculate Euclidean distance for negative samples (x_t and guiding_point)
        negative_distance = F.pairwise_distance(x_t_flat, guiding_point_flat)
        
        # Use norm of the model_output as the margin
        margin = 0.5
        
        # Compute positive contrastive loss
        positive_loss = th.mean((1 - margin) * th.pow(positive_distance, 2) +
                                margin * th.pow(th.clamp(margin - positive_distance, min=0), 2))
        
        # Compute negative contrastive loss
        negative_loss = th.mean((1 - margin) * th.pow(negative_distance, 2) +
                                margin * th.pow(th.clamp(margin - negative_distance, min=0), 2))
        
        # Total contrastive loss
        loss_contrastive = positive_loss + negative_loss
        
        return loss_contrastive

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t,  img, text, alpha, idx, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, img, self._scale_timesteps(t), text, alpha, idx, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        gt,
        t,
        img,
        text,
        alpha,
        idx,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            img,
            text,
            alpha,
            idx,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        gt,
        img,
        text,
        alpha,
        idx,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            gt,
            shape,
            img,
            text,
            alpha,
            idx,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        gt,
        shape,
        image,
        text,
        alpha,
        idx,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    gt,
                    t,
                    image,
                    text,
                    alpha,
                    idx,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, img, t, text, alpha, idx, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, img, self._scale_timesteps(t), text, alpha, idx, **model_kwargs)
        
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # Contrastive loss
            criterion = NCELoss()
            guiding_point = model.get_guiding_point()
            terms["contr"] = criterion(x_t, guiding_point, model_output)

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"] + 1e-3 * terms["contr"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

```

## image_datasets.py

```
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

```

## logger.py

```
"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def set_comm(comm):
    get_current().set_comm(comm)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def get_rank_without_mpi_import():
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def mpi_weighted_mean(comm, local_name2valcount):
    """
    Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger


```

## losses.py

```
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

```

## nn.py

```
"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

```

## resample.py

```
from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

```

## respace.py

```
import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, img, ts, text, alpha, idx, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, img, ts, text, alpha, idx, **kwargs)
    
    def get_guiding_point(self):
        return self.model.guiding_point

```

## script_util.py

```
import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

```

## train.util.py

```
import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

```

## unet.py

```
from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

```

# 2.hardware

## calibrate_camera.py

```
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from hardware.camera import RealSenseCamera


class Calibration:
    def __init__(self,
                 cam_id,
                 calib_grid_step,
                 checkerboard_offset_from_tool,
                 workspace_limits
                 ):
        self.calib_grid_step = calib_grid_step
        self.checkerboard_offset_from_tool = checkerboard_offset_from_tool

        # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.workspace_limits = workspace_limits

        self.camera = RealSenseCamera(device_id=cam_id)
        
        self.measured_pts = []
        self.observed_pts = []
        self.observed_pix = []
        self.camera2world = np.eye(4)

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        self.move_completed = os.path.join(homedir, "move_completed.npy")
        self.tool_position = os.path.join(homedir, "tool_position.npy")

    @staticmethod
    def _get_rigid_transform(A, B):
        """
        Estimate rigid transform with SVD (from Nghia Ho)
        """
        assert len(A) == len(B)

        N = A.shape[0]  # Total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
        BB = B - np.tile(centroid_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def _get_rigid_transform_error(self, z_scale):
        """
        Calculate the rigid transform RMS error

        :return RMS error
        """
        # Apply z offset and compute new observed points using camera intrinsics
        observed_z = np.squeeze(self.observed_pts[:, 2:] * z_scale)
        observed_x = np.multiply(np.squeeze(self.observed_pix[:, [0]]) - self.camera.intrinsics.ppx,
                                 observed_z / self.camera.intrinsics.fx)
        observed_y = np.multiply(np.squeeze(self.observed_pix[:, [1]]) - self.camera.intrinsics.ppy,
                                 observed_z / self.camera.intrinsics.fy)

        new_observed_pts = np.asarray([observed_x, observed_y, observed_z]).T

        # Estimate rigid transform between new observed points and measured points
        R, t = self._get_rigid_transform(np.asarray(new_observed_pts), np.asarray(self.measured_pts))
        t.shape = (3, 1)
        self.camera2world = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(new_observed_pts)) + np.tile(t, (1, new_observed_pts.shape[0]))
        error = np.transpose(registered_pts) - self.measured_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error / new_observed_pts.shape[0])
        return rmse

    def _generate_grid(self):
        """
        Construct 3D calibration grid across workspace

        :return calibration grid points
        """
        gridspace_x = np.linspace(self.workspace_limits[0][0], self.workspace_limits[0][1],
                                  1 + (self.workspace_limits[0][1] - self.workspace_limits[0][
                                      0]) / self.calib_grid_step)
        gridspace_y = np.linspace(self.workspace_limits[1][0], self.workspace_limits[1][1],
                                  1 + (self.workspace_limits[1][1] - self.workspace_limits[1][
                                      0]) / self.calib_grid_step)
        gridspace_z = np.linspace(self.workspace_limits[2][0], self.workspace_limits[2][1],
                                  1 + (self.workspace_limits[2][1] - self.workspace_limits[2][
                                      0]) / self.calib_grid_step)
        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts, 1)
        calib_grid_y.shape = (num_calib_grid_pts, 1)
        calib_grid_z.shape = (num_calib_grid_pts, 1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        return calib_grid_pts
        
    def run(self):
        # Connect to camera
        self.camera.connect()
        logging.debug(self.camera.intrinsics)

        logging.info('Collecting data...')

        calib_grid_pts = self._generate_grid()

        logging.info('Total grid points: ', calib_grid_pts.shape[0])

        for tool_position in calib_grid_pts:
            logging.info('Requesting move to tool position: ', tool_position)
            np.save(self.tool_position, tool_position)
            np.save(self.move_completed, 0)
            while not np.load(self.move_completed):
                time.sleep(0.1)
            # Wait for robot to be stable
            time.sleep(2)

            # Find checkerboard center
            checkerboard_size = (3, 3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_bundle = self.camera.get_image_bundle()
            camera_color_img = image_bundle['rgb']
            camera_depth_img = image_bundle['aligned_depth']
            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                    cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found:
                corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria)

                # Get observed checkerboard center 3D point in camera space
                checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)
                checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                checkerboard_x = np.multiply(checkerboard_pix[0] - self.camera.intrinsics.ppx,
                                             checkerboard_z / self.camera.intrinsics.fx)
                checkerboard_y = np.multiply(checkerboard_pix[1] - self.camera.intrinsics.ppy,
                                             checkerboard_z / self.camera.intrinsics.fy)
                if checkerboard_z == 0:
                    continue

                # Save calibration point and observed checkerboard center
                self.observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])
                # tool_position[2] += self.checkerboard_offset_from_tool
                tool_position = tool_position + self.checkerboard_offset_from_tool

                self.measured_pts.append(tool_position)
                self.observed_pix.append(checkerboard_pix)

                # Draw and display the corners
                vis = cv2.drawChessboardCorners(bgr_color_data, (1, 1), corners_refined[4, :, :], checkerboard_found)
                # cv2.imwrite('%06d.png' % len(self.measured_pts), vis)
                cv2.imshow('Calibration', vis)
                cv2.waitKey(10)
            else:
                logging.info('Checker board not found')

        self.measured_pts = np.asarray(self.measured_pts)
        self.observed_pts = np.asarray(self.observed_pts)
        self.observed_pix = np.asarray(self.observed_pix)

        # Optimize z scale w.r.t. rigid transform error
        logging.info('Calibrating...')
        z_scale_init = 1
        optim_result = optimize.minimize(self._get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
        camera_depth_offset = optim_result.x

        # Save camera optimized offset and camera pose
        logging.info('Saving...')
        np.savetxt('saved_data/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
        rmse = self._get_rigid_transform_error(camera_depth_offset)
        logging.info('RMSE: ', rmse)
        np.savetxt('saved_data/camera_pose.txt', self.camera2world, delimiter=' ')
        logging.info('Done.')

```

## camera.py

```
import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    while True:
        cam.plot_image_bundle()

```

## device.py

```
import logging

import torch

logging.basicConfig(level=logging.INFO)


def get_device(force_cpu):
    # Check if CUDA can be used
    if torch.cuda.is_available() and not force_cpu:
        logging.info("CUDA detected. Running with GPU acceleration.")
        device = torch.device("cuda")
    elif force_cpu:
        logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")
    return device

```

# 3.inference

## models

### clip_fusion

#### attention.py

```
import torch
import torch.nn as nn

import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, embed_dim, max_len=80):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        po = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.) / embed_dim))
        pe[:, 0::2] = torch.sin(po * div_term)
        pe[:, 1::2] = torch.cos(po * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        l, _, _ = x.shape
        pos = self.pe[:l, :].unsqueeze(1)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, embed_dim, max_len=80, padding_idx=0):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim, padding_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        l, _, _ = x.shape
        idx = torch.arange(l, device=x.device)
        pos = self.pos_embed(idx).unsqueeze(1)
        return pos

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
```

#### cross_attention.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from inference.models.clip_fusion.embedder import *
from inference.models.clip_fusion.attention import *

class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = CrossModalAttention(embed_dim=d_model, num_heads=n_head, output_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        attn_output, attn_weights = self.attn(q=q, k=k, v=v, attn_mask=self.attn_mask)
        return attn_output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn_output, attn_weights = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + attn_output
        q = q + self.mlp(self.ln_2(q))
        return q, attn_weights
    
# multi layer
class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([CrossResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        for i, _ in enumerate(self.resblocks):
            q, attn_weights = self.resblocks[i](q, k, v)

        q = q.permute(1, 0, 2) # L'ND -> NL'D
        return q, attn_weights

# one layer without shortcut: naivest cross attention
class CrossModalAttention(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, embed_dim=1024, num_heads=32, output_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, q, k, v, attn_mask=None):
        x, attn_weights = F.multi_head_attention_forward(
            query=q, key=k, value=v,
            embed_dim_to_check=v.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            need_weights=True,
            attn_mask=attn_mask
        )
        
        return x, attn_weights

```

#### embedder.py

```
import torch
import torch.nn as nn


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

# positional embedding with sin/cos
class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out
```

### lgdm

暂时略

### ragt

暂时略

### clipfusion.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from inference.models.clip_fusion.cross_attention import CrossTransformer
from inference.models.clip_fusion.embedder import get_embedder

from inference.models.ragt.mobile_vit import get_model
from inference.models.ragt.Anchor import *

from inference.models.grasp_model import LanguageGraspModel


class CLIPFusion(LanguageGraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, grasp_dim: int=5, width: int=3136, layers: int=6, heads: int=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.cross_attn = CrossTransformer(width=width, layers=layers, heads=heads).to(self.device)

        self.grasp_embbedding = nn.Sequential(
                                nn.Linear(grasp_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, width)
                                ).to(self.device)

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)
        
        self.bbox_pos_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, 256),
                                nn.GELU(),
                                nn.Linear(256, 1024),
                                nn.GELU(),
                                nn.Linear(1024, width)
                                ).to(self.device)
        
        self.text_embedding = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, width),
            nn.GELU(),
            nn.Linear(width, width)
        ).to(self.device)

        # Reshaped cosine layers
        self.conv_upsampling = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 3136)
        )

        # Reshaped linear layers
        self.upsampling_layer = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU()
        )

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        self._setup_pretrained_model()
    
    def _setup_pretrained_model(self):
        _default_weight_path = "weights/RAGT-3-3.pth"

        self.ragt_model = get_model()
        self.ragt_model.load_state_dict(torch.load(_default_weight_path))
        self.ragt_model.eval()
        for param in self.ragt_model.parameters():
            param.requires_grad = False


    def encode_grasp(self, x):
        grasp_emb = self.grasp_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_bbox_pos(self, x):
        bbox_pos_emb = self.bbox_pos_embbedding(x.to(self.device)) # shape = [N, L', D]
        return bbox_pos_emb
        
    
    def forward(self, bboxes, predicted_grasp, pos_bboxes, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text = clip.tokenize(text).to(device)

        image_features = self.clip_model.encode_image(bboxes)
        text_features = self.clip_model.encode_text(text)
        
        bbox_feat, text_feat = self.clip_model(bboxes, text)

        grasp_feat = self.encode_grasp(predicted_grasp)
        grasp_feat = grasp_feat.permute(1, 0, 2)

        # pos_bboxes = self.pos_projection(pos_bboxes)
        # bbox_pos_feat = self.encode_bbox_pos(pos_bboxes)
        # bbox_pos_feat = bbox_pos_feat.permute(1, 0, 2)

        bbox_pos_feat = torch.zeros(1, grasp_feat.shape[1], 3136).cuda()

        text_features = self.text_embedding(text_features.float())
        text_features = text_features.unsqueeze(1).permute(1, 0, 2)

        cross_feat, attn_weights = self.cross_attn(q=grasp_feat.float(), k=bbox_pos_feat.float(), v=text_features.float())

        # cross_feat = self.conv_upsampling(cross_feat)
        cross_feat = cross_feat.view(-1, 1, 56, 56).repeat(1, 128, 1, 1)
        x = cross_feat

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
    
    def _get_index_and_bias(self, output, confidence_threshold):
        N, C, H, W = output.shape
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(N, H, W, num_anchors, -1)
        mask_obj = (F.sigmoid(output[..., 0]) >= confidence_threshold)
        index = mask_obj.nonzero()
        bias = output[mask_obj]
        return index, bias

    def _get_coordinate(self, index, bias):
        confidence = torch.sigmoid(bias[:, 0])
        cx = (index[:, 2] + torch.sigmoid(bias[:, 1])) * field_of_grid_cell
        cy = (index[:, 1] + torch.sigmoid(bias[:, 2])) * field_of_grid_cell
        w = anchor_w * torch.exp(bias[:, 3])
        h = anchor_h * torch.exp(bias[:, 4])
        theta = (index[:, 3] + torch.sigmoid(bias[:, 5])) * theta_margin
        return confidence, cx, cy, w, h, theta
    
    def _predict_grasp_pose(self, x):
        output = self.ragt_model(x)
        index, bias = self._get_index_and_bias(output, 0.95)
        confidence, cx, cy, w, h, theta = self._get_coordinate(index, bias)
        return torch.cat([cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1), theta.unsqueeze(1)], dim=1)
    
    def compute_loss(self, xc, yc, prompt, query):
        y_pos, y_cos, y_sin, y_width = yc
        l = []
        for _xc in xc:
            _xc = _xc.unsqueeze(0)
            predicted_grasps = self._predict_grasp_pose(_xc)
            if predicted_grasps.data.shape[0] > 0:
                l.append(predicted_grasps[0].unsqueeze(0).unsqueeze(0))
            else:
                l.append(torch.zeros(1, 1, 5).cuda())
        predicted_grasps = torch.cat(l, dim=0)
        batch_size, num_grasps, _ = predicted_grasps.data.shape

        pos_bboxes = torch.zeros(batch_size, num_grasps, 3).cuda()
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, predicted_grasps, pos_bboxes, query)
        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

```

### ggcnn.py

```
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class GGCNN(GraspModel):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=5, output_padding=1)

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
```

### grasp_model.py

```
import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class LanguageGraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(LanguageGraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc, prompt, query):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, prompt, query)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

```

### grconvnet.py

```
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=1, dropout=False, prob=0.0, channel_size=32):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

        self.dropout1 = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        pos_output = self.pos_output(self.dropout1(x))
        cos_output = self.cos_output(self.dropout1(x))
        sin_output = self.sin_output(self.dropout1(x))
        width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

```

### grconvnet2.py

```
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

```

### grconvnet3.py

```
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

```

### grconvnet4.py

```
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size // 2)

        self.conv3 = nn.Conv2d(channel_size // 2, channel_size // 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size // 4)

        self.res1 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res2 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res3 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res4 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res5 = ResidualBlock(channel_size // 4, channel_size // 4)

        self.conv4 = nn.ConvTranspose2d(channel_size // 4, channel_size // 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size // 2)

        self.conv5 = nn.ConvTranspose2d(channel_size // 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

```

### lggcnn.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from inference.models.grasp_model import LanguageGraspModel

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class LGGCNN(LanguageGraspModel):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, clip_version='ViT-B/32'):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=5, output_padding=1)

        self.y_flatten = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 19),
            nn.GELU()
        )
        
        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        # Setup language modality
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x, prompt, query):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Encode text
        device = x.device
        y_feats = self._encode_text(query, device=device)
        y_feats = self.y_flatten(y_feats)
        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 19).unsqueeze(1).expand(-1, 8, -1, -1)

        # Combine textual features with the visual features
        x = torch.clone(x).detach() + y_feats

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc, prompt, query):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, prompt, query)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def _load_and_freeze_clip(self, clip_version, device=None):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def _encode_text(self, raw_text, device=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.lang_model.encode_text(texts).float()

```

### lgrconvnet3.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from inference.models.grasp_model import LanguageGraspModel, ResidualBlock


class GenerativeResnet(LanguageGraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, clip_version='ViT-B/32'):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.y_flatten = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 56),
            nn.GELU()
        )

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # Setup language modality
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in, prompt, query):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Encode text
        device = x.device
        y_feats = self._encode_text(query, device=device)
        y_feats = self.y_flatten(y_feats)
        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 56).unsqueeze(1).expand(-1, 128, -1, -1)
         
        # Combine textual features with the visual features
        x = torch.clone(x).detach() + y_feats
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def _load_and_freeze_clip(self, clip_version, device=None):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def _encode_text(self, raw_text, device=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.lang_model.encode_text(texts).float()
```

## grasp_generator.py

```
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp


class GraspGenerator:
    def __init__(self, saved_model_path, cam_id, visualize=False):
        self.saved_model_path = saved_model_path
        self.camera = RealSenseCamera(device_id=cam_id)

        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('saved_data/camera_depth_scale.txt', delimiter=' ')

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)

        # Get grasp position from model output
        pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)

        if pos_z == 0:
            return

        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        print('target: ', target)

        # Convert camera to robot coordinates
        camera2robot = self.cam_pose
        target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
        target_position = target_position[0:3, 0]

        # Convert camera to robot angle
        angle = np.asarray([0, 0, grasps[0].angle])
        angle.shape = (3, 1)
        target_angle = np.dot(camera2robot[0:3, 0:3], angle)

        # Concatenate grasp pose with grasp angle
        grasp_pose = np.append(target_position, target_angle[2])

        print('grasp_pose: ', grasp_pose)

        np.save(self.grasp_pose, grasp_pose)

        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)

    def run(self):
        while True:
            if np.load(self.grasp_request):
                self.generate()
                np.save(self.grasp_request, 0)
                np.save(self.grasp_available, 1)
            else:
                time.sleep(0.1)

```

## post_process.py

```
import torch
from skimage.filters import gaussian


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

```

# 4.models

暂时略

# 5.script

# 6.split

# 7.utils

## data

### camera_data.py

```
import numpy as np
import torch

from utils.dataset_processing import image


class CameraData:
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self,
                 width=640,
                 height=480,
                 output_size=224,
                 include_depth=True,
                 include_rgb=True
                 ):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        """
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_depth(self, img):
        depth_img = image.Image(img)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        depth_img.normalise()
        # depth_img.resize((self.output_size, self.output_size))
        depth_img.img = depth_img.img.transpose((2, 0, 1))
        return depth_img.img

    def get_rgb(self, img, norm=True):
        rgb_img = image.Image(img)
        rgb_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        # rgb_img.resize((self.output_size, self.output_size))
        if norm:
                rgb_img.normalise()
                rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_data(self, rgb=None, depth=None):
        depth_img = None
        rgb_img = None
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(img=depth)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(img=rgb)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         np.expand_dims(rgb_img, 0)),
                        1
                    )
                )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(np.expand_dims(rgb_img, 0))

        return x, depth_img, rgb_img

```

### cornell_data.py

```
import glob
import os

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

```

### grasp_anything_data.py

```
import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase


class GraspAnythingDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, 'positive_grasp', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'scene_description', '*.pkl'))
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        if kwargs["seen"]:
            with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))

        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        mask_file = self.grasp_files[idx].replace("positive_grasp", "mask").replace(".pt", ".npy")
        mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("positive_grasp", "image")
        rgb_img = image.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    def get_prompts(self, idx):
        prompt_file, obj_id = self.grasp_files[idx].replace("positive_grasp", "prompt").split('_')
        prompt_file += '.pkl'
        obj_id = int(obj_id.split('.')[0])

        with open(prompt_file, 'rb') as f:
            x = pickle.load(f)
            prompt, queries = x

        return prompt, queries[obj_id]
```

### grasp_anywhere_data.py

```
import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase


class GraspAnywhereDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything 数据集的主目录.
        :param ds_rotate: 数据集列表旋转比例（用于简单的交叉验证划分）
        :param kwargs: 其他关键字参数（包含 add_file_path, seen 等）
        """
        super(GraspAnywhereDataset, self).__init__(**kwargs)

        addition_file_path = kwargs["add_file_path"]

        self.grasp_files = glob.glob(os.path.join(addition_file_path, 'grasp_label_positive', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'scene_description', '*.pkl'))
        self.prompt_dir = os.path.join(file_path, 'scene_description')
        self.instruction_dir = os.path.join(addition_file_path, 'grasp_instructions')
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.rgb_dir = os.path.join(file_path, 'image')
        self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        if kwargs["seen"]:
            # 如果参数指定 seen=True (已知物体模式)
            with open(os.path.join('split/grasp-anything++/test/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)# 加载 seen.obj 中的 ID 列表
            # 过滤 grasp_files：只保留文件名（去掉最后5位字符后）存在于 idxs 列表中的文件
            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything++/test/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))


        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.mask_files.sort()

        # 抓取物体总数量
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            
    # 
    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    # 
    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    # 找到对应的 jpg 文件 -> 读取 -> 旋转/缩放增强 -> Resize 到 224x224 -> 转为 PyTorch 格式 (CHW)
    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        mask_file = self.grasp_files[idx].replace("positive_grasp", "mask").replace(".pt", ".npy")
        # mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.split('/')[-1]
        rgb_file = os.path.join(self.rgb_dir, rgb_file)
        rgb_img = image.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    
    def get_prompts(self, idx):
        # 1. 获取文件名
        grasp_file = self.grasp_files[idx].split('/')[-1]
        # 例如：self.grasp_files[idx] 是 ".../scene01_5_0.pt"
        # split('/')[-1] 拿到 "scene01_5_0.pt"

        # 2. 拆分文件名
        prompt_file, obj_id, part_id = grasp_file.split('_')
        # 这里假设文件名格式严格为： "场景名_物体ID_部件ID.pt"
        # 例如 "scene01_5_0.pt" 被拆分为：
        # prompt_file = "scene01" (场景对应的文件名)
        # obj_id = "5"            (物体索引)
        # part_id = "0.pt"        (抓取/部件索引)

        # 3. 构建指令文件路径 (但这部分后来没用)
        instruction_file = grasp_file.split('.')[0]
        instruction_file += '.pkl'
        instruction_file = os.path.join(self.instruction_dir, instruction_file)

        # 4. 构建 Prompt 文件路径 (这是真正用到的)
        prompt_file += '.pkl'# "scene01" -> "scene01.pkl"
        # 拼接完整路径，例如：".../scene_description/scene01.pkl"
        prompt_file = os.path.join(self.prompt_dir, prompt_file)
        obj_id = int(obj_id)
        part_id = int(part_id.split('.')[0])

        with open(prompt_file, 'rb') as f:
            x = pickle.load(f)
            prompt, queries = x
            # ('A small green apple and a yellow rubber duck sitting on a wooden table', ['apple', 'duck'])
        
        # 返回场景描述以及obj_id对应的物体
        return prompt, queries[obj_id]
```

### grasp_data.py

```
import random

import numpy as np
import torch
import torch.utils.data


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, seen=True):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)

```

### jacquard_data.py

```
import glob
import os

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class JacquardDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(JacquardDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', '*_grasps.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('perfect_depth.tiff', 'RGB.png') for f in self.depth_files]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])

```

### language_grasp_data.py

```
import random

import numpy as np
import torch
import torch.utils.data


class LanguageGraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, seen=True, add_file_path=None):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # Load the prompts
        prompt, query = self.get_prompts(idx)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor, prompt, query

    def __len__(self):
        return len(self.grasp_files)

```

## dataset_processing

### evaluation.py

```
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from .grasp import GraspRectangles, detect_grasps


def plot_output(fig, rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img:
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.pause(0.1)
    fig.canvas.draw()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False

```

### generate_cornell_depth.py

```
import argparse
import glob
import os

import numpy as np
from imageio import imsave

from utils.dataset_processing.image import DepthImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    parser.add_argument('path', type=str, help='Path to Cornell Grasping Dataset')
    args = parser.parse_args()

    pcds = glob.glob(os.path.join(args.path, '*', 'pcd*[0-9].txt'))
    pcds.sort()

    for pcd in pcds:
        di = DepthImage.from_pcd(pcd, (480, 640))
        di.inpaint()

        of_name = pcd.replace('.txt', 'd.tiff')
        print(of_name)
        imsave(of_name, di.img.astype(np.float32))

```

### grasp.py

```
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max


def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


def _grasp_anything_format(grasp: list):
    _, x, y, w, h, theta = grasp
    # index based on row, column (y,x), and the Grasp-Anything dataset's angles are flipped around an axis.
    return Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        
        return cls(grs)

    @classmethod
    def load_from_ocid_grasp_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        
        return cls(grs)

    @classmethod
    def load_from_vmrd_file(cls, fname):
        """
        Load grasp rectangles from a VMRD dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            grasp_lines = f.readlines()
            for grasp_line in grasp_lines:
                # x1, y1, x2, y2, x3, y3, x4, y4 = list(map(lambda x: int(round(float(x))), grasp_line.split(' ')[:8]))
                y1, x1, y2, x2, y3, x3, y4, x4 = list(map(lambda x: int(round(float(x))), grasp_line.split(' ')[:8]))
                try:
                    gr = np.array([
                        [x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4],
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    @classmethod
    def load_from_grasp_anything_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Grasp-Anything dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = None
        with open(fname, 'rb') as f:
            pos_grasps = torch.load(f)
        # add_fn = fname.replace("positive_grasp", "negative_grasp")
        # with open(fname, 'rb') as f:
        #     neg_grasps = torch.load(f)
        # grasps = torch.cat((pos_grasps, neg_grasps), dim=0).tolist()
        grasps = pos_grasps.tolist()
        grs = list(map(lambda x: _grasp_anything_format(x), grasps))

        grs = cls(grs)
        grs.scale(scale)
        return grs
    
    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle, self.length / 3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        ).astype(np.float))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * self.angle * 180 / np.pi, self.length * scale,
            self.width * scale)


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps

```

### image.py

```
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import imageio
from skimage.transform import rotate, resize

warnings.filterwarnings("ignore", category=UserWarning)


class Image:
    """
    Wrapper around an image with some convenient functions.
    """

    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname):
        return cls(imread(fname))

    def copy(self):
        """
        :return: Copy of self.
        """
        return self.__class__(self.img.copy())
    
    @classmethod
    def mask_out_image(cls, image, mask):
        # Apply the mask to the image
        masked_image = np.array(image)
        masked_image[:, :, 0] = masked_image[:, :, 0] * mask + 255 * (1-mask)
        masked_image[:, :, 1] = masked_image[:, :, 1] * mask + 255 * (1-mask)
        masked_image[:, :, 2] = masked_image[:, :, 2] * mask + 255 * (1-mask)

        return cls(imageio.core.util.Array(masked_image))

    def crop(self, top_left, bottom_right, resize=None):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        """
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        """
        :return: Cropped copy of the image.
        """
        i = self.copy()
        i.crop(*args, **kwargs)
        return i

    def normalise(self):
        """
        Normalise the image by converting to float [0,1] and zero-centering
        """
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

    def resize(self, shape):
        """
        Resize image to shape.
        :param shape: New shape.
        """
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)

    def resized(self, *args, **kwargs):
        """
        :return: Resized copy of the image.
        """
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        """
        Rotate the image.
        :param angle: Angle (in radians) to rotate by.
        :param center: Center pixel to rotate if specified, otherwise image center is used.
        """
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img, angle / np.pi * 180, center=center, mode='symmetric', preserve_range=True).astype(
            self.img.dtype)

    def rotated(self, *args, **kwargs):
        """
        :return: Rotated copy of image.
        """
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        """
        Plot the image
        :param ax: Existing matplotlib axis (optional)
        :param kwargs: kwargs to imshow
        """
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs):
        """
        :return: Zoomed copy of the image.
        """
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img):
        super().__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename, shape, default_filler=0, index=None):
        """
            Create a depth image from an unstructured PCD file.
            If index isn't specified, use euclidean distance, otherwise choose x/y/z=0/1/2
        """
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.split()

                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])

                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                else:
                    img[r, c] = float(ls[index])

        return cls(img / 1000.0)

    @classmethod
    def from_tiff(cls, fname):
        return cls(imread(fname))

    def inpaint(self, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        self.img = cv2.copyMakeBorder(self.img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (self.img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(self.img).max()
        self.img = self.img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        self.img = cv2.inpaint(self.img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        self.img = self.img[1:-1, 1:-1]
        self.img = self.img * scale

    def gradients(self):
        """
        Compute gradients of the depth image using Sobel filtesr.
        :return: Gradients in X direction, Gradients in Y diretion, Magnitude of XY gradients.
        """
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)

    def normalise(self):
        """
        Normalise by subtracting the mean and clippint [-1, 1]
        """
        self.img = np.clip((self.img - self.img.mean()), -1, 1)


class WidthImage(Image):
    """
    A width image is one that describes the desired gripper width at each pixel.
    """

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.  Also scales the width accordingly.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        super().zoom(factor)
        self.img = self.img / factor

    def normalise(self):
        """
        Normalise by mapping [0, 150] -> [0, 1]
        """
        self.img = np.clip(self.img, 0, 150.0) / 150.0

```

### mask.py

```
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

class Mask:
    """
    Wrapper around an mask with some convenient functions.
    """

    def __init__(self, mask):
        self.mask = mask

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.mask, attr)
    
    @classmethod
    def from_file(cls, fname):
        return np.load(fname, mmap_mode='r')
```

## visualisation

暂时略

# train_network.py

```
import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='lgrconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset') 
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere',
                        help='Specific for Grasp-Anywhere')
    
    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor, prompt, query in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc, prompt, query)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for x, y, _, _, _, prompt, query in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc, prompt, query)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      seen=args.seen,
                      add_file_path=args.add_file_path)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    )

    net = net.to(device)
    logging.info('Done')

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # Print model architecture.
    # summary(net, (input_channels, args.input_size, args.input_size))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, args.input_size, args.input_size))
    # sys.stdout = sys.__stdout__
    # f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
```

