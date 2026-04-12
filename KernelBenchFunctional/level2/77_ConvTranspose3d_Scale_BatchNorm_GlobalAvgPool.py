import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor', 'eps', 'momentum']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias', 'batch_norm_momentum', 'batch_norm_eps', 'global_avg_pool_output_size', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-05, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
    state_kwargs['batch_norm_momentum'] = model.batch_norm.momentum
    state_kwargs['batch_norm_eps'] = model.batch_norm.eps
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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
    batch_norm_running_mean,
    batch_norm_running_var,
    batch_norm_weight,
    batch_norm_bias,
    batch_norm_momentum,
    batch_norm_eps,
    global_avg_pool_output_size,
    scale_factor,
):
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias)
    x = x * scale_factor
    x = F.batch_norm(x, batch_norm_running_mean, batch_norm_running_var, batch_norm_weight, batch_norm_bias, training=False, momentum=batch_norm_momentum, eps=batch_norm_eps)
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    return x
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

