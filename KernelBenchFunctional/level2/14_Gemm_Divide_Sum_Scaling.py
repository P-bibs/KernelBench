import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['input_size', 'hidden_size', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['weight', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['weight']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

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
    if 'weight' in flat_state:
        state_kwargs['weight'] = flat_state['weight']
    else:
        state_kwargs['weight'] = getattr(model, 'weight')
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
    weight,
    scaling_factor,
):
    x = torch.matmul(x, weight.T)
    x = x / 2
    x = torch.sum(x, dim=1, keepdim=True)
    x = x * scaling_factor
    return x
batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]


def get_inputs():
    return [torch.rand(batch_size, input_size)]

