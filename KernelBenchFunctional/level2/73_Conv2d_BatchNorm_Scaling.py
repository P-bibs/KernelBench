import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'bn_momentum', 'bn_eps', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, *args) -> torch.Tensor:
        return functional_model(*args, **extract_state_kwargs(self))


def build_reference_model():
    init_inputs = list(get_init_inputs())
    model = ModelNew(*init_inputs)
    model.eval()
    return model


def extract_state_kwargs(model):
    flat_state = {}
    for name, value in model.named_parameters():
        flat_state[name.replace('.', '_')] = value
    for name, value in model.named_buffers():
        flat_state[name.replace('.', '_')] = value
    state_kwargs = {}
    init_inputs = list(get_init_inputs())
    init_arg_map = {name: value for name, value in zip(INIT_PARAM_NAMES, init_inputs)}
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    # State for bn (nn.BatchNorm2d)
    if 'bn_running_mean' in flat_state:
        state_kwargs['bn_running_mean'] = flat_state['bn_running_mean']
    else:
        state_kwargs['bn_running_mean'] = getattr(model.bn, 'running_mean', None)
    if 'bn_running_var' in flat_state:
        state_kwargs['bn_running_var'] = flat_state['bn_running_var']
    else:
        state_kwargs['bn_running_var'] = getattr(model.bn, 'running_var', None)
    if 'bn_weight' in flat_state:
        state_kwargs['bn_weight'] = flat_state['bn_weight']
    else:
        state_kwargs['bn_weight'] = getattr(model.bn, 'weight', None)
    if 'bn_bias' in flat_state:
        state_kwargs['bn_bias'] = flat_state['bn_bias']
    else:
        state_kwargs['bn_bias'] = getattr(model.bn, 'bias', None)
    state_kwargs['bn_momentum'] = model.bn.momentum
    state_kwargs['bn_eps'] = model.bn.eps
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs


# END EVAL UTILS

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    bn_running_mean,
    bn_running_var,
    bn_weight,
    bn_bias,
    bn_momentum,
    bn_eps,
    scaling_factor,
):
    x = F.conv2d(x, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False, momentum=bn_momentum, eps=bn_eps)
    x = x * scaling_factor
    return x
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

