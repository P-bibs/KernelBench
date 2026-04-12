import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias', 'batch_norm_momentum', 'batch_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'batch_norm_running_mean', 'batch_norm_running_var', 'batch_norm_weight', 'batch_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

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
    # State for batch_norm (nn.BatchNorm1d)
    if 'batch_norm_running_mean' in flat_state:
        state_kwargs['batch_norm_running_mean'] = flat_state['batch_norm_running_mean']
    else:
        state_kwargs['batch_norm_running_mean'] = getattr(model.batch_norm, 'running_mean', None)
    if 'batch_norm_running_var' in flat_state:
        state_kwargs['batch_norm_running_var'] = flat_state['batch_norm_running_var']
    else:
        state_kwargs['batch_norm_running_var'] = getattr(model.batch_norm, 'running_var', None)
    if 'batch_norm_weight' in flat_state:
        state_kwargs['batch_norm_weight'] = flat_state['batch_norm_weight']
    else:
        state_kwargs['batch_norm_weight'] = getattr(model.batch_norm, 'weight', None)
    if 'batch_norm_bias' in flat_state:
        state_kwargs['batch_norm_bias'] = flat_state['batch_norm_bias']
    else:
        state_kwargs['batch_norm_bias'] = getattr(model.batch_norm, 'bias', None)
    state_kwargs['batch_norm_momentum'] = model.batch_norm.momentum
    state_kwargs['batch_norm_eps'] = model.batch_norm.eps
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
    batch_norm_running_mean,
    batch_norm_running_var,
    batch_norm_weight,
    batch_norm_bias,
    batch_norm_momentum,
    batch_norm_eps,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = F.batch_norm(x, batch_norm_running_mean, batch_norm_running_var, batch_norm_weight, batch_norm_bias, training=False, momentum=batch_norm_momentum, eps=batch_norm_eps)
    x = torch.nn.functional.gelu(x)
    x = torch.relu(x)
    return x
batch_size = 16384
in_features = 4096
out_features = 4096


def get_init_inputs():
    return [in_features, out_features]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

