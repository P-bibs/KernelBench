import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

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
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = x * torch.sigmoid(x)
    x = x / 2.0
    x = torch.clamp(x, min=-1.0, max=1.0)
    x = torch.tanh(x)
    x = torch.clamp(x, min=-1.0, max=1.0)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192


def get_init_inputs():
    return [in_features, out_features]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

