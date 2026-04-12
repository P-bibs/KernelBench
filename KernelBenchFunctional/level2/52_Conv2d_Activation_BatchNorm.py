import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'eps', 'momentum']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'bn_momentum', 'bn_eps']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-05, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

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
    bn_running_mean,
    bn_running_var,
    bn_weight,
    bn_bias,
    bn_momentum,
    bn_eps,
):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False, momentum=bn_momentum, eps=bn_eps)
    return x
batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

