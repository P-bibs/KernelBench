import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'multiplier', 'negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'leaky_relu_negative_slope', 'multiplier']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

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
    # State for leaky_relu (nn.LeakyReLU)
    state_kwargs['leaky_relu_negative_slope'] = model.leaky_relu.negative_slope
    if 'multiplier' in flat_state:
        state_kwargs['multiplier'] = flat_state['multiplier']
    else:
        state_kwargs['multiplier'] = getattr(model, 'multiplier')
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
    leaky_relu_negative_slope,
    multiplier,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = x * multiplier
    x = F.leaky_relu(x, negative_slope=leaky_relu_negative_slope)
    return x
batch_size = 1024
in_features  = 8192  
out_features = 8192
multiplier = 2.0
negative_slope = 0.1


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

