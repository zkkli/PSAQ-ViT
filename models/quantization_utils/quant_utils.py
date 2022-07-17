import math
import numpy as np
from torch.autograd import Function, Variable
import torch


def reshape_tensor(input, scale, zero_point, is_weight=True):
    if is_weight:
        if len(input.shape) == 4:
            range_shape = (-1, 1, 1, 1)
        elif len(input.shape) == 2:
            range_shape = (-1, 1)
        else:
            raise NotImplementedError
    else:
        if len(input.shape) == 2:
            range_shape = (1, -1)
        elif len(input.shape) == 3:
            range_shape = (1, 1, -1)
        elif len(input.shape) == 4:
            range_shape = (1, -1, 1, 1)
        else:
            raise NotImplementedError

    scale = scale.reshape(range_shape)
    zero_point = zero_point.reshape(range_shape)

    return scale, zero_point


def symmetric_linear_quantization_params(num_bits,
                                         min_val,
                                         max_val):
    """
    Compute the scaling factor and zeropoint with the given quantization range for symmetric quantization.
    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    eps = torch.finfo(torch.float32).eps

    max_val = torch.max(-min_val, max_val)
    scale = max_val / (float(qmax - qmin) / 2)
    scale.clamp_(eps)
    zero_point = torch.zeros_like(max_val, dtype=torch.int64)

    return scale, zero_point, qmin, qmax


def asymmetric_linear_quantization_params(num_bits,
                                          min_val,
                                          max_val):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    qmax = 2 ** num_bits - 1
    qmin = 0
    eps = torch.finfo(torch.float32).eps

    scale = (max_val - min_val) / float(qmax - qmin)
    scale.clamp_(eps)
    zero_point = qmin - torch.round(min_val / scale)
    zero_point.clamp_(qmin, qmax)

    return scale, zero_point, qmin, qmax


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """
        scale, zero_point, qmin, qmax = symmetric_linear_quantization_params(k, x_min, x_max)
        scale, zero_point = reshape_tensor(x, scale, zero_point, is_weight=True)

        # quantize
        quant_x = x / scale + zero_point
        quant_x = quant_x.round().clamp(qmin, qmax)

        # dequantize
        quant_x = (quant_x - zero_point) * scale

        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """
        scale, zero_point, qmin, qmax = asymmetric_linear_quantization_params(k, x_min, x_max)
        scale, zero_point = reshape_tensor(x, scale, zero_point, is_weight=False)

        # quantize
        quant_x = x / scale + zero_point
        quant_x = quant_x.round().clamp(qmin, qmax)
        
        # dequantize
        quant_x = (quant_x - zero_point) * scale

        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
