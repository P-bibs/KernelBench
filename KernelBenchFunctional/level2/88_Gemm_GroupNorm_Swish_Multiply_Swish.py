import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'num_groups', 'multiply_weight_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'multiply_weight']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'group_norm_weight', 'group_norm_bias', 'multiply_weight']


class ModelNew(nn.Module):
    """
    ModelNew that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """

    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

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
    # State for gemm (nn.Linear)
    if 'gemm_weight' in flat_state:
        state_kwargs['gemm_weight'] = flat_state['gemm_weight']
    else:
        state_kwargs['gemm_weight'] = getattr(model.gemm, 'weight', None)
    if 'gemm_bias' in flat_state:
        state_kwargs['gemm_bias'] = flat_state['gemm_bias']
    else:
        state_kwargs['gemm_bias'] = getattr(model.gemm, 'bias', None)
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
    if 'multiply_weight' in flat_state:
        state_kwargs['multiply_weight'] = flat_state['multiply_weight']
    else:
        state_kwargs['multiply_weight'] = getattr(model, 'multiply_weight')
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
    gemm_weight,
    gemm_bias,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    multiply_weight,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = F.group_norm(x, group_norm_num_groups, group_norm_weight, group_norm_bias)
    x = x * torch.sigmoid(x)
    x = x * multiply_weight
    x = x * torch.sigmoid(x)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

