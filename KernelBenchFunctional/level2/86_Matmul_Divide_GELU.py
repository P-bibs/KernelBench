import torch
import torch.nn as nn
import torch.nn.functional as F

# BEGIN EVAL UTILS
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['input_size', 'output_size', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """

    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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
    linear_weight,
    linear_bias,
    divisor,
):
    x = F.linear(x, linear_weight, linear_bias)
    x = x / divisor
    x = torch.nn.functional.gelu(x)
    return x
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_init_inputs():
    return [input_size, output_size, divisor]


def get_inputs():
    return [torch.rand(batch_size, input_size)]

