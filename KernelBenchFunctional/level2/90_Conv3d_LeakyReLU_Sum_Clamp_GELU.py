import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'sum_tensor_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'sum_tensor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'sum_tensor']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    if 'sum_tensor' in flat_state:
        state_kwargs['sum_tensor'] = flat_state['sum_tensor']
    else:
        state_kwargs['sum_tensor'] = getattr(model, 'sum_tensor')
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
    sum_tensor,
):
    x = F.conv3d(x, conv_weight, conv_bias)
    x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    x = x + sum_tensor
    x = torch.clamp(x, min=-1.0, max=1.0)
    x = torch.nn.functional.gelu(x)
    return x
batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

