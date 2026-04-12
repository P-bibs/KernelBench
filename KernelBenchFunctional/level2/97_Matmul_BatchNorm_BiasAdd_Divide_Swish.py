import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'bn_eps', 'bn_momentum', 'bias_shape', 'divide_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['matmul_weight', 'matmul_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'bn_momentum', 'bn_eps', 'bias', 'divide_value']
REQUIRED_FLAT_STATE_NAMES = ['matmul_weight', 'matmul_bias', 'bn_running_mean', 'bn_running_var', 'bn_weight', 'bn_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """

    def __init__(self, in_features, out_features, bn_eps=1e-05, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

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
    # State for matmul (nn.Linear)
    if 'matmul_weight' in flat_state:
        state_kwargs['matmul_weight'] = flat_state['matmul_weight']
    else:
        state_kwargs['matmul_weight'] = getattr(model.matmul, 'weight', None)
    if 'matmul_bias' in flat_state:
        state_kwargs['matmul_bias'] = flat_state['matmul_bias']
    else:
        state_kwargs['matmul_bias'] = getattr(model.matmul, 'bias', None)
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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'divide_value' in flat_state:
        state_kwargs['divide_value'] = flat_state['divide_value']
    else:
        state_kwargs['divide_value'] = getattr(model, 'divide_value')
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
    matmul_weight,
    matmul_bias,
    bn_running_mean,
    bn_running_var,
    bn_weight,
    bn_bias,
    bn_momentum,
    bn_eps,
    bias,
    divide_value,
):
    x = F.linear(x, matmul_weight, matmul_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False, momentum=bn_momentum, eps=bn_eps)
    x = x + bias
    x = x / divide_value
    x = x * torch.sigmoid(x)
    return x
batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]


def get_inputs():
    return [torch.rand(batch_size, in_features)]

