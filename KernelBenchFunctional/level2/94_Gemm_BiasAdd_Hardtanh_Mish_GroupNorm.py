import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bias_shape', 'num_groups']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'groupnorm_weight', 'groupnorm_bias', 'groupnorm_num_groups', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'groupnorm_weight', 'groupnorm_bias', 'bias']


class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """

    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

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
    # State for hardtanh (nn.Hardtanh)
    # State for mish (nn.Mish)
    # State for groupnorm (nn.GroupNorm)
    if 'groupnorm_weight' in flat_state:
        state_kwargs['groupnorm_weight'] = flat_state['groupnorm_weight']
    else:
        state_kwargs['groupnorm_weight'] = getattr(model.groupnorm, 'weight', None)
    if 'groupnorm_bias' in flat_state:
        state_kwargs['groupnorm_bias'] = flat_state['groupnorm_bias']
    else:
        state_kwargs['groupnorm_bias'] = getattr(model.groupnorm, 'bias', None)
    state_kwargs['groupnorm_num_groups'] = model.groupnorm.num_groups
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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
    groupnorm_weight,
    groupnorm_bias,
    groupnorm_num_groups,
    bias,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = x + bias
    x = F.hardtanh(x)
    x = F.mish(x)
    x = F.group_norm(x, groupnorm_num_groups, groupnorm_weight, groupnorm_bias)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256


def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

