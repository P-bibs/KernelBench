import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'scaling_factor', 'hardtanh_min', 'hardtanh_max']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'hardtanh_min_val', 'hardtanh_max_val', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a GEMM, scaling, hardtanh, and GELU activation.
    """

    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
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
    state_kwargs['hardtanh_min_val'] = model.hardtanh.min_val
    state_kwargs['hardtanh_max_val'] = model.hardtanh.max_val
    # State for gelu (nn.GELU)
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
    gemm_weight,
    gemm_bias,
    hardtanh_min_val,
    hardtanh_max_val,
    scaling_factor,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = x * scaling_factor
    x = F.hardtanh(x, min_val=hardtanh_min_val, max_val=hardtanh_max_val)
    x = F.gelu(x)
    return x
batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2


def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

