import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'groups', 'min_value', 'max_value', 'dropout_p']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = ['min_value', 'max_value']
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'norm_weight', 'norm_bias', 'norm_num_groups', 'dropout_p', 'min_value', 'max_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'norm_weight', 'norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

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
    # State for norm (nn.GroupNorm)
    if 'norm_weight' in flat_state:
        state_kwargs['norm_weight'] = flat_state['norm_weight']
    else:
        state_kwargs['norm_weight'] = getattr(model.norm, 'weight', None)
    if 'norm_bias' in flat_state:
        state_kwargs['norm_bias'] = flat_state['norm_bias']
    else:
        state_kwargs['norm_bias'] = getattr(model.norm, 'bias', None)
    state_kwargs['norm_num_groups'] = model.norm.num_groups
    # State for dropout (nn.Dropout)
    state_kwargs['dropout_p'] = model.dropout.p
    if 'min_value' in init_arg_map:
        state_kwargs['min_value'] = init_arg_map['min_value']
    else:
        state_kwargs['min_value'] = min_value
    if 'max_value' in init_arg_map:
        state_kwargs['max_value'] = init_arg_map['max_value']
    else:
        state_kwargs['max_value'] = max_value
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
    norm_weight,
    norm_bias,
    norm_num_groups,
    dropout_p,
    min_value,
    max_value,
):
    x = F.conv3d(x, conv_weight, conv_bias)
    x = F.group_norm(x, norm_num_groups, norm_weight, norm_bias)
    x = torch.min(x, torch.tensor(min_value, device=x.device))
    x = torch.clamp(x, min=min_value, max=max_value)
    x = F.dropout(x, training=False, p=dropout_p)
    return x
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

