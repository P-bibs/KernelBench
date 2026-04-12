import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'sum_weight', 'norm_shape', 'pool_kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'norm_weight', 'norm_bias', 'norm_normalized_shape', 'avg_pool_kernel_size', 'sum_weight']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'norm_weight', 'norm_bias', 'sum_weight']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

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
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    # State for norm (nn.LayerNorm)
    if 'norm_weight' in flat_state:
        state_kwargs['norm_weight'] = flat_state['norm_weight']
    else:
        state_kwargs['norm_weight'] = getattr(model.norm, 'weight', None)
    if 'norm_bias' in flat_state:
        state_kwargs['norm_bias'] = flat_state['norm_bias']
    else:
        state_kwargs['norm_bias'] = getattr(model.norm, 'bias', None)
    state_kwargs['norm_normalized_shape'] = model.norm.normalized_shape
    # State for avg_pool (nn.AvgPool3d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    # State for gelu (nn.GELU)
    if 'sum_weight' in flat_state:
        state_kwargs['sum_weight'] = flat_state['sum_weight']
    else:
        state_kwargs['sum_weight'] = getattr(model, 'sum_weight')
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
    conv_transpose_output_padding,
    norm_weight,
    norm_bias,
    norm_normalized_shape,
    avg_pool_kernel_size,
    sum_weight,
):
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding, output_padding=conv_transpose_output_padding)
    x = x + sum_weight
    x = F.layer_norm(x, norm_normalized_shape, norm_weight, norm_bias)
    x = F.avg_pool3d(x, kernel_size=avg_pool_kernel_size)
    x = F.gelu(x)
    return x
batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

