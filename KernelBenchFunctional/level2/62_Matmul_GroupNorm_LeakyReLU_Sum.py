import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['input_size', 'hidden_size', 'num_groups', 'eps', 'negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['fc_weight', 'fc_bias', 'gn_weight', 'gn_bias', 'gn_num_groups', 'gn_eps', 'leaky_relu_negative_slope', 'leaky_relu_inplace']
REQUIRED_FLAT_STATE_NAMES = ['fc_weight', 'fc_bias', 'gn_weight', 'gn_bias']


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """

    def __init__(self, input_size, hidden_size, num_groups, eps=1e-05, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

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
    # State for fc (nn.Linear)
    if 'fc_weight' in flat_state:
        state_kwargs['fc_weight'] = flat_state['fc_weight']
    else:
        state_kwargs['fc_weight'] = getattr(model.fc, 'weight', None)
    if 'fc_bias' in flat_state:
        state_kwargs['fc_bias'] = flat_state['fc_bias']
    else:
        state_kwargs['fc_bias'] = getattr(model.fc, 'bias', None)
    # State for gn (nn.GroupNorm)
    if 'gn_weight' in flat_state:
        state_kwargs['gn_weight'] = flat_state['gn_weight']
    else:
        state_kwargs['gn_weight'] = getattr(model.gn, 'weight', None)
    if 'gn_bias' in flat_state:
        state_kwargs['gn_bias'] = flat_state['gn_bias']
    else:
        state_kwargs['gn_bias'] = getattr(model.gn, 'bias', None)
    state_kwargs['gn_num_groups'] = model.gn.num_groups
    state_kwargs['gn_eps'] = model.gn.eps
    # State for leaky_relu (nn.LeakyReLU)
    state_kwargs['leaky_relu_negative_slope'] = model.leaky_relu.negative_slope
    state_kwargs['leaky_relu_inplace'] = model.leaky_relu.inplace
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
    fc_weight,
    fc_bias,
    gn_weight,
    gn_bias,
    gn_num_groups,
    gn_eps,
    leaky_relu_negative_slope,
    leaky_relu_inplace,
):
    x = F.linear(x, fc_weight, fc_bias)
    x = F.group_norm(x, gn_num_groups, gn_weight, gn_bias, eps=gn_eps)
    x = F.leaky_relu(x, negative_slope=leaky_relu_negative_slope, inplace=leaky_relu_inplace)
    x = x + x
    return x
batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


def get_init_inputs():
    return [input_size, hidden_size, num_groups]


def get_inputs():
    return [torch.rand(batch_size, input_size)]

