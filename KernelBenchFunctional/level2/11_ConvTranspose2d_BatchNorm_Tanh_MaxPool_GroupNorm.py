import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'groups', 'num_groups']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias', 'max_pool_kernel_size', 'max_pool_stride', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias', 'group_norm_weight', 'group_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    # State for batch_norm (nn.BatchNorm2d)
    if 'batch_norm_running_mean' in flat_state:
        state_kwargs['batch_norm_running_mean'] = flat_state['batch_norm_running_mean']
    else:
        state_kwargs['batch_norm_running_mean'] = getattr(model.batch_norm, 'running_mean', None)
    if 'batch_norm_running_var' in flat_state:
        state_kwargs['batch_norm_running_var'] = flat_state['batch_norm_running_var']
    else:
        state_kwargs['batch_norm_running_var'] = getattr(model.batch_norm, 'running_var', None)
    if 'batch_norm_weight' in flat_state:
        state_kwargs['batch_norm_weight'] = flat_state['batch_norm_weight']
    else:
        state_kwargs['batch_norm_weight'] = getattr(model.batch_norm, 'weight', None)
    if 'batch_norm_bias' in flat_state:
        state_kwargs['batch_norm_bias'] = flat_state['batch_norm_bias']
    else:
        state_kwargs['batch_norm_bias'] = getattr(model.batch_norm, 'bias', None)
    # State for tanh (nn.Tanh)
    # State for max_pool (nn.MaxPool2d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
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
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    batch_norm_running_mean,
    batch_norm_running_var,
    batch_norm_weight,
    batch_norm_bias,
    max_pool_kernel_size,
    max_pool_stride,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
):
    x = F.conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding)
    x = F.batch_norm(x, batch_norm_running_mean, batch_norm_running_var, batch_norm_weight, batch_norm_bias, training=False)
    x = torch.tanh(x)
    x = F.max_pool2d(x, kernel_size=max_pool_kernel_size, stride=max_pool_stride)
    x = F.group_norm(x, group_norm_num_groups, group_norm_weight, group_norm_bias)
    return x
batch_size = 512
in_channels  = 64  
out_channels = 128  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

