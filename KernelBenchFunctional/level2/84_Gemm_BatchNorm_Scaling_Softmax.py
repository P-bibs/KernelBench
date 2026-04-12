import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bn_eps', 'bn_momentum', 'scale_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'bn_momentum', 'bn_eps', 'softmax_dim', 'scale']
REQUIRED_FLAT_STATE_NAMES = ['gemm_weight', 'gemm_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'scale']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """

    def __init__(self, in_features, out_features, bn_eps=1e-05, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
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
    softmax_dim,
    scale,
):
    x = F.linear(x, gemm_weight, gemm_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False, momentum=bn_momentum, eps=bn_eps)
    x = scale * x
    x = F.softmax(x, dim=softmax_dim)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

