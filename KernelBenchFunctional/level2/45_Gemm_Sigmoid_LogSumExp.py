import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['input_size', 'hidden_size', 'output_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear1_weight', 'linear1_bias', 'linear2_weight', 'linear2_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear1_weight', 'linear1_bias', 'linear2_weight', 'linear2_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

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
    # State for linear1 (nn.Linear)
    if 'linear1_weight' in flat_state:
        state_kwargs['linear1_weight'] = flat_state['linear1_weight']
    else:
        state_kwargs['linear1_weight'] = getattr(model.linear1, 'weight', None)
    if 'linear1_bias' in flat_state:
        state_kwargs['linear1_bias'] = flat_state['linear1_bias']
    else:
        state_kwargs['linear1_bias'] = getattr(model.linear1, 'bias', None)
    # State for linear2 (nn.Linear)
    if 'linear2_weight' in flat_state:
        state_kwargs['linear2_weight'] = flat_state['linear2_weight']
    else:
        state_kwargs['linear2_weight'] = getattr(model.linear2, 'weight', None)
    if 'linear2_bias' in flat_state:
        state_kwargs['linear2_bias'] = flat_state['linear2_bias']
    else:
        state_kwargs['linear2_bias'] = getattr(model.linear2, 'bias', None)
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
    linear1_weight,
    linear1_bias,
    linear2_weight,
    linear2_bias,
):
    x = F.linear(x, linear1_weight, linear1_bias)
    x = torch.sigmoid(x)
    x = F.linear(x, linear2_weight, linear2_bias)
    x = torch.logsumexp(x, dim=1)
    return x
batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024


def get_init_inputs():
    return [input_size, hidden_size, output_size]


def get_inputs():
    return [torch.rand(batch_size, input_size)]

