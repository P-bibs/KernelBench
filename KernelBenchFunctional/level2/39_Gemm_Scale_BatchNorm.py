import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'scale_shape', 'eps', 'momentum']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'bn_momentum', 'bn_eps', 'scale']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'scale']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """

    def __init__(self, in_features, out_features, scale_shape, eps=1e-05, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

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
    # State for bn (nn.BatchNorm1d)
    if 'bn_running_mean' in flat_state:
        state_kwargs['bn_running_mean'] = flat_state['bn_running_mean']
    else:
        state_kwargs['bn_running_mean'] = getattr(model.bn, 'running_mean', None)
    if 'bn_running_var' in flat_state:
        state_kwargs['bn_running_var'] = flat_state['bn_running_var']
    else:
        state_kwargs['bn_running_var'] = getattr(model.bn, 'running_var', None)
    if 'bn_weight' in flat_state:
        state_kwargs['bn_weight'] = flat_state['bn_weight']
    else:
        state_kwargs['bn_weight'] = getattr(model.bn, 'weight', None)
    if 'bn_bias' in flat_state:
        state_kwargs['bn_bias'] = flat_state['bn_bias']
    else:
        state_kwargs['bn_bias'] = getattr(model.bn, 'bias', None)
    state_kwargs['bn_momentum'] = model.bn.momentum
    state_kwargs['bn_eps'] = model.bn.eps
    if 'scale' in flat_state:
        state_kwargs['scale'] = flat_state['scale']
    else:
        state_kwargs['scale'] = getattr(model, 'scale')
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
    bn_running_mean,
    bn_running_var,
    bn_weight,
    bn_bias,
    bn_momentum,
    bn_eps,
    scale,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = x * scale
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False, momentum=bn_momentum, eps=bn_eps)
    return x
batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)


def get_init_inputs():
    return [in_features, out_features, scale_shape]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

