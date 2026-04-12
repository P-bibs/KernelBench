import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias', 'avg_pool1_kernel_size', 'avg_pool2_kernel_size']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias']


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    # State for batch_norm (nn.BatchNorm3d)
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
    # State for avg_pool1 (nn.AvgPool3d)
    state_kwargs['avg_pool1_kernel_size'] = model.avg_pool1.kernel_size
    # State for avg_pool2 (nn.AvgPool3d)
    state_kwargs['avg_pool2_kernel_size'] = model.avg_pool2.kernel_size
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
    avg_pool1_kernel_size,
    avg_pool2_kernel_size,
):
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding)
    x = F.batch_norm(x, batch_norm_running_mean, batch_norm_running_var, batch_norm_weight, batch_norm_bias, training=False)
    x = F.avg_pool3d(x, kernel_size=avg_pool1_kernel_size)
    x = F.avg_pool3d(x, kernel_size=avg_pool2_kernel_size)
    return x
batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

