import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias', 'eps', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'layer_norm_weight', 'layer_norm_bias', 'layer_norm_normalized_shape', 'layer_norm_eps', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'layer_norm_weight', 'layer_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-05, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
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
    # State for layer_norm (nn.LayerNorm)
    if 'layer_norm_weight' in flat_state:
        state_kwargs['layer_norm_weight'] = flat_state['layer_norm_weight']
    else:
        state_kwargs['layer_norm_weight'] = getattr(model.layer_norm, 'weight', None)
    if 'layer_norm_bias' in flat_state:
        state_kwargs['layer_norm_bias'] = flat_state['layer_norm_bias']
    else:
        state_kwargs['layer_norm_bias'] = getattr(model.layer_norm, 'bias', None)
    state_kwargs['layer_norm_normalized_shape'] = model.layer_norm.normalized_shape
    state_kwargs['layer_norm_eps'] = model.layer_norm.eps
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
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    layer_norm_weight,
    layer_norm_bias,
    layer_norm_normalized_shape,
    layer_norm_eps,
    scaling_factor,
):
    x = F.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, stride=conv_transpose_stride, padding=conv_transpose_padding)
    x = F.layer_norm(x, layer_norm_normalized_shape, layer_norm_weight, layer_norm_bias, eps=layer_norm_eps)
    x = torch.nn.functional.gelu(x)
    x = x * scaling_factor
    return x
batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

