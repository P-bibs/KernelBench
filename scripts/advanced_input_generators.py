from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBLEM_ROOT = REPO_ROOT / "KernelBench"


SIGNED_ACTIVATION_STRESS = "signed_activation_stress"
SOFTMAX_EXTREME_LOGITS_STRESS = "softmax_extreme_logits_stress"
MAXPOOL_ARG_REDUCE_TIE_STRESS = "maxpool_arg_reduce_tie_stress"
AXIS_SIGNATURE_NORM_STRESS = "axis_signature_norm_stress"
QUASI_STRUCTURED_MATMUL_STRESS = "quasi_structured_matmul_stress"
SCAN_EDGE_VALUE_STRESS = "scan_edge_value_stress"
SDPA_PATTERN_STRESS = "sdpa_pattern_stress"


Generator = Callable[[ModuleType], list[torch.Tensor | object]]


def _clone_inputs(inputs: list[torch.Tensor | object]) -> list[torch.Tensor | object]:
    cloned: list[torch.Tensor | object] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.detach().clone())
        else:
            cloned.append(item)
    return cloned


def _get_base_inputs(module: ModuleType) -> list[torch.Tensor | object]:
    return _clone_inputs(module.get_inputs())


def _promote_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(dtype=torch.float32)
    return tensor


def _restore_dtype(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.is_floating_point():
        return tensor.to(dtype=reference.dtype)
    return tensor


def _inject_small_signed_sentinels(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor

    step = max(1, flat.numel() // 4096)
    sentinel_values = [0.0, 1e-7, -1e-7, 0.5, -0.5, 4.0, -4.0]
    for offset, value in enumerate(sentinel_values):
        index = min(flat.numel() - 1, offset * step)
        flat[index] = value
    return tensor


def signed_activation_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return inputs

    reference = inputs[0]
    x = torch.randn_like(_promote_float_tensor(reference)) * 4.0
    x = _inject_small_signed_sentinels(x)
    inputs[0] = _restore_dtype(x, reference)
    return inputs


def softmax_extreme_logits_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return inputs

    reference = inputs[0]
    x = torch.full_like(_promote_float_tensor(reference), -40.0)

    if x.ndim >= 2:
        reduce_axis = 1
        axis_size = x.shape[reduce_axis]
        outer = x.shape[0]
        for batch_idx in range(outer):
            hot0 = batch_idx % axis_size
            hot1 = (batch_idx + 1) % axis_size
            selectors = [slice(None)] * x.ndim
            selectors[0] = batch_idx
            selectors[reduce_axis] = hot0
            x[tuple(selectors)] = 40.0
            selectors[reduce_axis] = hot1
            x[tuple(selectors)] = 39.75
    else:
        x.fill_(-80.0)
        if x.numel() >= 2:
            x[0] = 80.0
            x[1] = 79.75

    inputs[0] = _restore_dtype(x, reference)
    return inputs


def maxpool_arg_reduce_tie_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return inputs

    relpath = _relative_problem_path(module)
    reference = inputs[0]
    x = -5.0 - torch.rand_like(_promote_float_tensor(reference))

    wants_min = "Argmin" in relpath or "Min_" in relpath or "_Min_" in relpath
    low_value = -20.0
    high_value = 20.0
    tie_value = low_value if wants_min else high_value

    if x.ndim >= 2:
        selectors0 = [slice(None)] * x.ndim
        selectors1 = [slice(None)] * x.ndim
        selectors0[1] = 0
        selectors1[1] = 1 if x.shape[1] > 1 else 0
        x[tuple(selectors0)] = tie_value
        x[tuple(selectors1)] = tie_value

    for dim in range(2, x.ndim):
        if x.shape[dim] > 1:
            leading = [slice(None)] * x.ndim
            trailing = [slice(None)] * x.ndim
            leading[dim] = 0
            trailing[dim] = 1
            x[tuple(leading)] = tie_value
            x[tuple(trailing)] = tie_value

    inputs[0] = _restore_dtype(x, reference)
    return inputs


def axis_signature_norm_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return inputs

    reference = inputs[0]
    x = torch.zeros_like(_promote_float_tensor(reference))

    for dim, size in enumerate(x.shape):
        shape = [1] * x.ndim
        shape[dim] = size
        coord = torch.arange(size, dtype=x.dtype, device=x.device).reshape(shape)
        weight = 10.0 ** max(0, x.ndim - dim - 1)
        x = x + weight * coord

    x = x + 1e-3 * torch.randn_like(x)
    inputs[0] = _restore_dtype(x, reference)
    return inputs


def quasi_structured_matmul_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if len(inputs) < 2 or not isinstance(inputs[0], torch.Tensor) or not isinstance(inputs[1], torch.Tensor):
        return inputs

    relpath = _relative_problem_path(module)
    ref_a = inputs[0]
    ref_b = inputs[1]
    a = _promote_float_tensor(ref_a)
    b = _promote_float_tensor(ref_b)

    if "symmetric" in relpath.lower():
        a = a + 1e-2 * torch.triu(torch.randn_like(a), diagonal=1)
        b = b - 1e-2 * torch.triu(torch.randn_like(b), diagonal=1)
    elif "upper_triangular" in relpath:
        a = a + 1e-2 * torch.tril(torch.randn_like(a), diagonal=-1)
        b = b - 1e-2 * torch.tril(torch.randn_like(b), diagonal=-1)
    elif "lower_triangular" in relpath:
        a = a + 1e-2 * torch.triu(torch.randn_like(a), diagonal=1)
        b = b - 1e-2 * torch.triu(torch.randn_like(b), diagonal=1)
    elif "transposed_A" in relpath:
        a = a + 1e-2 * torch.randn_like(a)
    elif "transposed_B" in relpath:
        b = b + 1e-2 * torch.randn_like(b)
    elif "transposed_both" in relpath:
        a = a + 1e-2 * torch.randn_like(a)
        b = b - 1e-2 * torch.randn_like(b)

    inputs[0] = _restore_dtype(a, ref_a)
    inputs[1] = _restore_dtype(b, ref_b)
    return inputs


def scan_edge_value_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return inputs

    reference = inputs[0]
    x = torch.zeros_like(_promote_float_tensor(reference))
    dim = getattr(module, "dim", 1)
    length = x.shape[dim]

    pattern = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 1e-7, -1e-7], dtype=x.dtype, device=x.device)
    reps = (length + pattern.numel() - 1) // pattern.numel()
    tiled = pattern.repeat(reps)[:length]
    view_shape = [1] * x.ndim
    view_shape[dim] = length
    x.copy_(tiled.reshape(view_shape).expand_as(x))

    if "cumprod" in _relative_problem_path(module):
        x.select(dim, 0).fill_(1.0)
        if length > 3:
            x.select(dim, 3).fill_(0.0)

    inputs[0] = _restore_dtype(x, reference)

    if len(inputs) > 1 and isinstance(inputs[1], torch.Tensor) and inputs[1].dtype == torch.bool:
        mask = torch.zeros_like(inputs[1], dtype=torch.bool)
        third = max(1, length // 3)
        mask.narrow(dim, 0, third).fill_(False)
        mask.narrow(dim, third, min(third, length - third)).fill_(True)
        if length > 2 * third:
            mask.narrow(dim, 2 * third, length - 2 * third).fill_(False)
        inputs[1] = mask

    return inputs


def sdpa_pattern_stress(module: ModuleType) -> list[torch.Tensor | object]:
    inputs = _get_base_inputs(module)
    if len(inputs) != 3 or not all(isinstance(item, torch.Tensor) for item in inputs):
        return inputs

    q_ref, k_ref, v_ref = inputs  # type: ignore[misc]
    q = torch.zeros_like(_promote_float_tensor(q_ref))
    k = torch.zeros_like(_promote_float_tensor(k_ref))
    v = torch.zeros_like(_promote_float_tensor(v_ref))

    _, _, seq_len, embed_dim = q.shape
    token_ids = torch.arange(seq_len, device=q.device)
    primary = token_ids % embed_dim
    secondary = (token_ids * 17 + 1) % embed_dim

    for idx in range(seq_len):
        q[:, :, idx, primary[idx]] = 20.0
        k[:, :, idx, primary[idx]] = 20.0
        v[:, :, idx, primary[idx]] = 1.0
        v[:, :, idx, secondary[idx]] = float(idx + 1) / float(seq_len)

    return [
        _restore_dtype(q, q_ref),
        _restore_dtype(k, k_ref),
        _restore_dtype(v, v_ref),
    ]


ADVANCED_INPUT_GENERATORS: dict[str, Generator] = {
    SIGNED_ACTIVATION_STRESS: signed_activation_stress,
    SOFTMAX_EXTREME_LOGITS_STRESS: softmax_extreme_logits_stress,
    MAXPOOL_ARG_REDUCE_TIE_STRESS: maxpool_arg_reduce_tie_stress,
    AXIS_SIGNATURE_NORM_STRESS: axis_signature_norm_stress,
    QUASI_STRUCTURED_MATMUL_STRESS: quasi_structured_matmul_stress,
    SCAN_EDGE_VALUE_STRESS: scan_edge_value_stress,
    SDPA_PATTERN_STRESS: sdpa_pattern_stress,
}


SIGNED_ACTIVATION_PROBLEMS = {
    "level1/19_ReLU.py",
    "level1/20_LeakyReLU.py",
    "level1/21_Sigmoid.py",
    "level1/22_Tanh.py",
    "level1/25_Swish.py",
    "level1/26_GELU_.py",
    "level1/27_SELU_.py",
    "level1/28_HardSigmoid.py",
    "level1/29_Softplus.py",
    "level1/30_Softsign.py",
    "level1/31_ELU.py",
    "level1/32_HardTanh.py",
    "level1/36_RMSNorm_.py",
    "level1/38_L1Norm_.py",
    "level1/39_L2Norm_.py",
    "level1/88_MinGPTNewGelu.py",
}


QUASI_STRUCTURED_MATMUL_PROBLEMS = {
    "level1/13_Matmul_for_symmetric_matrices.py",
    "level1/14_Matmul_for_upper_triangular_matrices.py",
    "level1/15_Matmul_for_lower_triangular_matrices.py",
    "level1/16_Matmul_with_transposed_A.py",
    "level1/17_Matmul_with_transposed_B.py",
    "level1/18_Matmul_with_transposed_both.py",
}


SCAN_PROBLEMS = {
    "level1/89_cumsum.py",
    "level1/90_cumprod.py",
    "level1/91_cumsum_reverse.py",
    "level1/92_cumsum_exclusive.py",
    "level1/93_masked_cumsum.py",
}


SDPA_PROBLEMS = {
    "level1/97_ScaledDotProductAttention.py",
}


def _relative_problem_path(module: ModuleType) -> str:
    module_path = Path(module.__file__).resolve()
    return module_path.relative_to(PROBLEM_ROOT).as_posix()


def _problem_relpaths() -> list[str]:
    return sorted(path.relative_to(PROBLEM_ROOT).as_posix() for path in PROBLEM_ROOT.rglob("*.py"))


def _problem_text(relpath: str) -> str:
    return (PROBLEM_ROOT / relpath).read_text(encoding="utf-8")


def _classify_problem(relpath: str) -> frozenset[str]:
    generators: set[str] = set()
    text = _problem_text(relpath)

    if relpath in SIGNED_ACTIVATION_PROBLEMS:
        generators.add(SIGNED_ACTIVATION_STRESS)

    if relpath in QUASI_STRUCTURED_MATMUL_PROBLEMS:
        generators.add(QUASI_STRUCTURED_MATMUL_STRESS)

    if relpath in SCAN_PROBLEMS:
        generators.add(SCAN_EDGE_VALUE_STRESS)

    if relpath in SDPA_PROBLEMS:
        generators.add(SDPA_PATTERN_STRESS)

    if any(token in text for token in ("BatchNorm", "InstanceNorm", "GroupNorm", "LayerNorm", "RMSNorm")):
        generators.add(AXIS_SIGNATURE_NORM_STRESS)

    if any(token in text for token in ("softmax", "Softmax", "log_softmax", "logsumexp", "CrossEntropyLoss", "KLDivLoss")):
        generators.add(SOFTMAX_EXTREME_LOGITS_STRESS)

    if any(
        token in text
        for token in (
            "MaxPool",
            "Max_Pooling",
            "argmax",
            "argmin",
            "torch.max(",
            "torch.min(",
            "Max_reduction",
            "Min_reduction",
        )
    ):
        generators.add(MAXPOOL_ARG_REDUCE_TIE_STRESS)

    return frozenset(sorted(generators))


def build_problem_to_advanced_inputs() -> dict[str, frozenset[str]]:
    mapping: OrderedDict[str, frozenset[str]] = OrderedDict()
    for relpath in _problem_relpaths():
        mapping[relpath] = _classify_problem(relpath)
    return dict(mapping)


PROBLEM_TO_ADVANCED_INPUTS = build_problem_to_advanced_inputs()


def apply_advanced_input_generator(module: ModuleType, generator_name: str) -> list[torch.Tensor | object]:
    generator = ADVANCED_INPUT_GENERATORS[generator_name]
    return generator(module)
