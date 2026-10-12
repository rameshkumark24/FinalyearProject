"""Net1D, the network of ECGFounder (Li et al., NEJM AI 2025; arXiv:2410.04133).

A port of ``net1d.py`` from the official repository (github.com/PKUDigitalHealth/ECGFounder,
written by Shenda Hong, 2020; the model card on Hugging Face states the MIT licence). Module and
parameter names are kept exactly as in the original so that the published checkpoint loads with
``strict=True``: if the port drifted from the original, loading fails instead of running with
partly random weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """Conv1d with "same" padding: output length = ceil(input length / stride)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1) -> None:
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        out_length = (length + self.stride - 1) // self.stride
        pad = max(0, (out_length - 1) * self.stride + self.kernel_size - length)
        return self.conv(F.pad(x, (pad // 2, pad - pad // 2), "constant", 0))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = max(0, self.kernel_size - 1)
        return self.max_pool(F.pad(x, (pad // 2, pad - pad // 2), "constant", 0))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """conv1 -> grouped conv k -> conv1, squeeze-and-excitation, residual connection."""

    def __init__(self, in_channels: int, out_channels: int, ratio: float, kernel_size: int, stride: int, groups: int,
                 downsample: bool, is_first_block: bool = False, use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block, self.use_bn, self.use_do = is_first_block, use_bn, use_do
        middle = int(out_channels * ratio)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, middle, kernel_size=1, stride=1, groups=1)

        self.bn2 = nn.BatchNorm1d(middle)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(middle, middle, kernel_size=kernel_size, stride=self.stride, groups=groups)

        self.bn3 = nn.BatchNorm1d(middle)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(middle, out_channels, kernel_size=1, stride=1, groups=1)

        self.se_fc1 = nn.Linear(out_channels, out_channels // 2)
        self.se_fc2 = nn.Linear(out_channels // 2, out_channels)
        self.se_activation = Swish()
        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)

        se = torch.sigmoid(self.se_fc2(self.se_activation(self.se_fc1(out.mean(-1)))))
        out = torch.einsum("abc,ab->abc", out, se)

        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            first = (self.out_channels - self.in_channels) // 2
            identity = F.pad(identity, (first, self.out_channels - self.in_channels - first), "constant", 0)
            identity = identity.transpose(-1, -2)
        return out + identity


class BasicStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: float, kernel_size: int, stride: int, groups: int,
                 i_stage: int, m_blocks: int, use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        self.block_list = nn.ModuleList()
        for i_block in range(m_blocks):
            first = i_block == 0
            self.block_list.append(BasicBlock(
                in_channels=in_channels if first else out_channels, out_channels=out_channels, ratio=ratio,
                kernel_size=kernel_size, stride=stride if first else 1, groups=groups, downsample=first,
                is_first_block=(i_stage == 0 and first), use_bn=use_bn, use_do=use_do))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        return x


class Net1D(nn.Module):
    """Input (batch, channels, samples). ``forward`` returns (logits, features)."""

    def __init__(self, in_channels: int, base_filters: int, ratio: float, filter_list: list[int],
                 m_blocks_list: list[int], kernel_size: int, stride: int, groups_width: int, n_classes: int,
                 use_bn: bool = True, use_do: bool = True) -> None:
        super().__init__()
        self.use_bn = use_bn
        self.first_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size=kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()
        self.stage_list = nn.ModuleList()
        channels = base_filters
        for i_stage, (out_channels, m_blocks) in enumerate(zip(filter_list, m_blocks_list)):
            self.stage_list.append(BasicStage(channels, out_channels, ratio, kernel_size, stride,
                                              groups=out_channels // groups_width, i_stage=i_stage, m_blocks=m_blocks,
                                              use_bn=use_bn, use_do=use_do))
            channels = out_channels
        self.dense = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.first_conv(x)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        for stage in self.stage_list:
            out = stage(out)
        features = out.mean(-1)
        return self.dense(features), features


def ecgfounder_single_lead(n_classes: int) -> Net1D:
    """The single-lead configuration of ``ft_1lead_ECGFounder`` in the official ``finetune_model.py``."""
    return Net1D(in_channels=1, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                 m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16,
                 n_classes=n_classes, use_bn=False, use_do=False)
