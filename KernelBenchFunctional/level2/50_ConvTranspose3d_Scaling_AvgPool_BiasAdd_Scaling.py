import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'scale1', 'scale2', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'avg_pool_kernel_size', 'scale1', 'bias', 'scale2']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'scale1', 'bias', 'scale2']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

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
    # State for avg_pool (nn.AvgPool3d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    if 'scale1' in flat_state:
        state_kwargs['scale1'] = flat_state['scale1']
    else:
        state_kwargs['scale1'] = getattr(model, 'scale1')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'scale2' in flat_state:
        state_kwargs['scale2'] = flat_state['scale2']
    else:
        state_kwargs['scale2'] = getattr(model, 'scale2')
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
    avg_pool_kernel_size,
    scale1,
    bias,
    scale2,
):
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding)
    x = x * scale1
    x = F.avg_pool3d(x, kernel_size=avg_pool_kernel_size)
    x = x + bias
    x = x * scale2
    return x
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

