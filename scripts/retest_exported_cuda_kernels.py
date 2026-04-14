#!/usr/bin/env python3
"""
Test AI-CUDA-Engineer archive dataset with advanced generators.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int = 0, initial: int = 0, desc: str | None = None, unit: str | None = None):
            self.total = total
            self.n = initial
            self.desc = desc
            self.unit = unit

        def update(self, amount: int):
            self.n += amount

        def close(self):
            return None


SCRIPT_DIR = Path(__file__).resolve().parent
KERNELBENCH_ROOT = SCRIPT_DIR.parent
REPO_ROOT = KERNELBENCH_ROOT.parent
SRC_DIR = KERNELBENCH_ROOT / "src"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_PROBLEM_IDS = (6, 24, 43, 57, 61, 66, 73, 84, 89, 96, 99, 100)
DEFAULT_EXPORT_ROOT = Path("/tmp/level2_export_test")
DEFAULT_RESULTS_ROOT = KERNELBENCH_ROOT / "results" / "retest_exported_cuda"

TASK_STATUS_PASSED = "passed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_TIMEOUT = "timeout"
TASK_STATUS_ERROR = "error"
TASK_STATUS_INTERRUPTED = "interrupted"

FINAL_STATUS_PASSES_ALL = "passes_base_and_all_advanced"
FINAL_STATUS_FAILS_BASE_ONLY = "fails_base_only"
FINAL_STATUS_FAILS_ADVANCED_ONLY = "fails_advanced_only"
FINAL_STATUS_FAILS_BOTH = "fails_both"
FINAL_STATUS_SKIPPED_NO_ADVANCED = "skipped_no_advanced_generator"
FINAL_STATUS_INCOMPLETE = "incomplete"

BASE_GENERATOR_NAME = "base"
EVENTS_FILENAME = "events.jsonl"
SUMMARY_JSON_FILENAME = "summary.json"
SUMMARY_CSV_FILENAME = "summary.csv"
MISMATCH_SUMMARY_FILENAME = "mismatch_summary.txt"

_STOP_REQUESTED = False


def ensure_executable_dir_on_path():
    executable_dir = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    entries = current_path.split(os.pathsep) if current_path else []
    if executable_dir not in entries:
        os.environ["PATH"] = executable_dir if not current_path else f"{executable_dir}{os.pathsep}{current_path}"


ensure_executable_dir_on_path()


@dataclass(frozen=True)
class BindingParam:
    name: str
    kind: str
    raw_type: str


@dataclass(frozen=True)
class CandidateSpec:
    task_id: int
    export_problem_dir: str
    export_problem_name: str
    kernel_name: str
    json_path: str
    cu_path: str
    reference_path: str
    reference_relpath: str
    generator_names: tuple[str, ...]
    initially_correct: bool
    initial_max_diff: float | None
    initial_error: str | None
    init_arg_names: tuple[str, ...]
    forward_arg_names: tuple[str, ...]
    binding_name: str
    binding_params: tuple[BindingParam, ...]

    @property
    def solution_key(self) -> str:
        return f"{self.task_id}/{self.export_problem_dir}/{self.kernel_name}"


@dataclass(frozen=True)
class EvalTask:
    task_key: str
    solution_key: str
    generator_name: str
    gpu_id: int
    build_dir: str
    candidate: CandidateSpec
    num_correct_trials: int
    backend: str
    precision: str


@dataclass(frozen=True)
class DiscoveryStats:
    total_scanned: int
    eligible: int
    skipped_missing_cu: int
    skipped_status_filtered: int
    skipped_unparseable_cuda: int
    ignored_non_json: int


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    return str(value)


def parse_problem_ids(raw: str) -> tuple[int, ...]:
    ids: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        ids.append(int(piece))
    if not ids:
        raise ValueError("Problem list is empty.")
    return tuple(dict.fromkeys(ids))


def parse_gpu_ids(raw: str | None) -> list[int]:
    if raw is None:
        import torch

        count = torch.cuda.device_count()
        if count <= 0:
            raise ValueError("No visible CUDA devices found. Pass --gpus explicitly or run on a CUDA host.")
        return list(range(count))

    gpu_ids: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        gpu_ids.append(int(piece))
    if not gpu_ids:
        raise ValueError("GPU list is empty after parsing --gpus.")
    return gpu_ids


def parse_export_problem_dir_name(path: Path) -> tuple[int, str] | None:
    match = re.match(r"^\d+_(\d+)_(.+)$", path.name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def normalize_numeric_diff(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return None


def should_include_candidate(metadata: dict[str, Any]) -> bool:
    correct = metadata.get("Correct")
    if correct is True:
        return True
    if correct is False and normalize_numeric_diff(metadata.get("Max_Diff")) is not None:
        return True
    return False


def resolve_reference_problem(task_id: int, kernelbench_root: Path) -> Path:
    candidate_dir = kernelbench_root / "KernelBench" / "level2"
    matches = sorted(candidate_dir.glob(f"{task_id}_*.py"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one level2 KernelBench reference for task {task_id}, found {len(matches)}.")
    return matches[0]


def load_problem_to_advanced_inputs() -> dict[str, frozenset[str]]:
    from advanced_input_generators import PROBLEM_TO_ADVANCED_INPUTS

    return PROBLEM_TO_ADVANCED_INPUTS


def extract_model_signature(problem_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse(problem_path.read_text(encoding="utf-8"), filename=str(problem_path))
    model_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            model_class = node
            break
    if model_class is None:
        raise ValueError(f"{problem_path} does not define Model.")

    init_names: tuple[str, ...] | None = None
    forward_names: tuple[str, ...] | None = None
    for node in model_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_names = tuple(arg.arg for arg in node.args.args[1:])
        elif isinstance(node, ast.FunctionDef) and node.name == "forward":
            forward_names = tuple(arg.arg for arg in node.args.args[1:])

    if init_names is None or forward_names is None:
        raise ValueError(f"{problem_path} is missing __init__ or forward on Model.")
    return init_names, forward_names


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def _extract_signature_text(source: str, function_name: str) -> str:
    cleaned = _strip_c_comments(source)
    match = re.search(r"\b(?:torch::Tensor|at::Tensor|void)\s+" + re.escape(function_name) + r"\s*\(", cleaned)
    if match is None:
        raise ValueError(f"Could not locate {function_name}(...) binding implementation in CUDA source.")

    start = cleaned.find("(", match.start())
    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return cleaned[start + 1 : index]
    raise ValueError(f"Unbalanced parentheses while parsing {function_name}(...) signature.")


def parse_binding_spec(cu_path: Path) -> tuple[str, tuple[BindingParam, ...]]:
    source = cu_path.read_text(encoding="utf-8")
    binding_match = re.search(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*&([A-Za-z_][A-Za-z0-9_]*)', source)
    if binding_match is not None:
        exported_name = binding_match.group(1)
        impl_name = binding_match.group(2)
    else:
        exported_name = "forward"
        impl_name = "forward"

    signature = _extract_signature_text(source, impl_name)
    params: list[BindingParam] = []
    for raw_piece in signature.split(","):
        piece = " ".join(raw_piece.strip().split())
        if not piece:
            continue
        match = re.match(r"(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)$", piece)
        if match is None:
            raise ValueError(f"Unable to parse parameter from forward(...) signature: {raw_piece!r}")
        raw_type = match.group(1).strip()
        name = match.group(2).strip()
        kind = "tensor" if "Tensor" in raw_type else "scalar"
        params.append(BindingParam(name=name, kind=kind, raw_type=raw_type))
    if not params:
        raise ValueError(f"No parameters parsed from forward(...) in {cu_path}")
    return exported_name, tuple(params)


def build_reference_source(problem_path: Path, generator_name: str) -> str:
    if generator_name == BASE_GENERATOR_NAME:
        return problem_path.read_text(encoding="utf-8")

    problem_path_str = json.dumps(str(problem_path.resolve()))
    script_dir_str = json.dumps(str(SCRIPT_DIR.resolve()))
    module_name = f"_kb_problem_{abs(hash((str(problem_path.resolve()), generator_name)))}"
    module_name_str = json.dumps(module_name)
    generator_name_str = json.dumps(generator_name)

    return f"""import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = {script_dir_str}
if _SCRIPTS_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPTS_DIR)

from advanced_input_generators import apply_advanced_input_generator as _apply_advanced_input_generator

_PROBLEM_PATH = _Path({problem_path_str})
_GENERATOR_NAME = {generator_name_str}


def _load_problem_module():
    spec = _importlib_util.spec_from_file_location({module_name_str}, _PROBLEM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load KernelBench problem module from {{_PROBLEM_PATH}}")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROBLEM_MODULE = _load_problem_module()
Model = _PROBLEM_MODULE.Model


def get_init_inputs():
    return _PROBLEM_MODULE.get_init_inputs()


def get_inputs():
    return _apply_advanced_input_generator(_PROBLEM_MODULE, _GENERATOR_NAME)
"""


def _extension_name_for_task(task: EvalTask) -> str:
    digest = hashlib.sha1(task.task_key.encode("utf-8")).hexdigest()[:12]
    return f"kb_export_ext_{digest}"


def build_custom_wrapper_source(task: EvalTask) -> str:
    candidate = task.candidate
    cu_source = Path(candidate.cu_path).read_text(encoding="utf-8")
    problem_path = Path(candidate.reference_path)

    problem_path_str = json.dumps(str(problem_path.resolve()))
    module_name = f"_kb_ref_{abs(hash((candidate.solution_key, task.generator_name)))}"
    ext_name = _extension_name_for_task(task)

    init_names = json.dumps(list(candidate.init_arg_names))
    forward_names = json.dumps(list(candidate.forward_arg_names))
    binding_params = json.dumps([asdict(param) for param in candidate.binding_params])
    cu_source_json = json.dumps(cu_source)
    module_name_json = json.dumps(module_name)
    ext_name_json = json.dumps(ext_name)
    cublas_needed = "cublas" in cu_source.lower()

    return f"""import importlib.util as _importlib_util
import json as _json
import sys as _sys
from pathlib import Path as _Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_PROBLEM_PATH = _Path({problem_path_str})
_MODULE_NAME = {module_name_json}
_INIT_ARG_NAMES = {init_names}
_FORWARD_ARG_NAMES = {forward_names}
_BINDING_NAME = {json.dumps(candidate.binding_name)}
_BINDING_PARAMS = _json.loads({json.dumps(binding_params)})
_CUDA_SOURCE = {cu_source_json}


def _load_problem_module():
    spec = _importlib_util.spec_from_file_location(_MODULE_NAME, _PROBLEM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load KernelBench problem module from {{_PROBLEM_PATH}}")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROBLEM_MODULE = _load_problem_module()
_EXTRA_LDFLAGS = {['-lcublas'] if cublas_needed else []}
_EXT = load_inline(
    name={ext_name_json},
    cpp_sources="",
    cuda_sources=_CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=_EXTRA_LDFLAGS,
    with_cuda=True,
    verbose=False,
)


def _is_simple_value(value):
    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_simple_value(item) for item in value)
    return False


def _add_value_if_simple(values, key, value):
    if key not in values and _is_simple_value(value):
        values[key] = value


def _collect_value_map(model, forward_args):
    values = {{}}
    for name, value in zip(_FORWARD_ARG_NAMES, forward_args):
        values[name] = value

    init_map = getattr(model, "_kb_init_arg_map", {{}})
    values.update(init_map)

    for name, value in model.named_parameters():
        values[name.replace(".", "_")] = value
    for name, value in model.named_buffers():
        values[name.replace(".", "_")] = value

    for name, value in vars(model).items():
        if name.startswith("_"):
            continue
        _add_value_if_simple(values, name, value)

    for module_name, module in model.named_modules():
        if not module_name:
            continue
        _add_value_if_simple(values, module_name, module)
        for attr_name in (
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "dilation",
            "groups",
            "eps",
            "momentum",
            "dim",
            "p",
            "negative_slope",
            "weight",
            "bias",
            "running_mean",
            "running_var",
        ):
            if hasattr(module, attr_name):
                _add_value_if_simple(values, f"{{module_name}}_{{attr_name}}", getattr(module, attr_name))

    return values


def _candidate_names(name):
    aliases = [name]
    if name.endswith("_tensor"):
        aliases.append(name[: -len("_tensor")])
    explicit_aliases = {{
        "gemm_weight": ["linear_weight", "matmul_weight"],
        "gemm_bias": ["linear_bias", "matmul_bias"],
        "linear_weight": ["gemm_weight", "matmul_weight"],
        "linear_bias": ["gemm_bias", "matmul_bias"],
        "matmul_weight": ["linear_weight", "gemm_weight"],
        "matmul_bias": ["linear_bias", "gemm_bias"],
        "scale": ["scaling_factor"],
        "dropout_p": ["p"],
        "p": ["dropout_p"],
        "weight": ["bn_weight", "group_norm_weight"],
        "bias": ["bn_bias", "group_norm_bias"],
        "running_mean": ["bn_running_mean"],
        "running_var": ["bn_running_var"],
    }}
    aliases.extend(explicit_aliases.get(name, []))
    return aliases


def _convert_scalar_value(value, name):
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value
    return value


def _resolve_binding_value(values, name, kind):
    seen = set()
    for candidate_name in _candidate_names(name):
        if candidate_name in seen:
            continue
        seen.add(candidate_name)
        if candidate_name in values:
            value = values[candidate_name]
            return _convert_scalar_value(value, name) if kind == "scalar" else value

    suffix_matches = [key for key in values if key.endswith("_" + name)]
    suffix_priority = [
        f"bn_{{name}}",
        f"group_norm_{{name}}",
        f"gemm_{{name}}",
        f"linear_{{name}}",
        f"matmul_{{name}}",
        f"conv_{{name}}",
        f"conv_transpose_{{name}}",
        f"max_pool_{{name}}",
    ]
    for preferred in suffix_priority:
        if preferred in suffix_matches:
            value = values[preferred]
            return _convert_scalar_value(value, name) if kind == "scalar" else value
    if len(suffix_matches) == 1:
        value = values[suffix_matches[0]]
        return _convert_scalar_value(value, name) if kind == "scalar" else value

    available = ", ".join(sorted(values.keys()))
    raise KeyError(f"Unable to resolve binding parameter '{{name}}' (kind={{kind}}). Available keys: {{available}}")


class ModelNew(_PROBLEM_MODULE.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kb_init_arg_map = {{name: value for name, value in zip(_INIT_ARG_NAMES, args)}}

    def forward(self, *args):
        values = _collect_value_map(self, args)
        call_args = []
        for param in _BINDING_PARAMS:
            call_args.append(_resolve_binding_value(values, param["name"], param["kind"]))
        return getattr(_EXT, _BINDING_NAME)(*call_args)
"""


def discover_candidates(export_root: Path, problem_ids: tuple[int, ...]) -> tuple[list[CandidateSpec], DiscoveryStats]:
    problem_id_set = set(problem_ids)
    advanced_inputs = load_problem_to_advanced_inputs()
    candidates: list[CandidateSpec] = []

    total_scanned = 0
    skipped_missing_cu = 0
    skipped_status_filtered = 0
    skipped_unparseable_cuda = 0
    ignored_non_json = 0

    for problem_dir in sorted(export_root.iterdir()):
        if not problem_dir.is_dir():
            continue

        parsed = parse_export_problem_dir_name(problem_dir)
        if parsed is None:
            continue
        task_id, export_problem_name = parsed
        if task_id not in problem_id_set:
            continue

        reference_path = resolve_reference_problem(task_id, KERNELBENCH_ROOT)
        reference_relpath = reference_path.relative_to(KERNELBENCH_ROOT / "KernelBench").as_posix()
        generator_names = tuple(sorted(advanced_inputs.get(reference_relpath, frozenset())))
        init_arg_names, forward_arg_names = extract_model_signature(reference_path)

        for item in sorted(problem_dir.iterdir()):
            if item.suffix != ".json":
                ignored_non_json += 1
                continue

            total_scanned += 1
            metadata = json.loads(item.read_text(encoding="utf-8"))
            cu_path = item.with_suffix(".cu")
            if not cu_path.exists():
                skipped_missing_cu += 1
                continue
            if not should_include_candidate(metadata):
                skipped_status_filtered += 1
                continue

            try:
                binding_name, binding_params = parse_binding_spec(cu_path)
            except ValueError:
                skipped_unparseable_cuda += 1
                continue
            candidates.append(
                CandidateSpec(
                    task_id=task_id,
                    export_problem_dir=problem_dir.name,
                    export_problem_name=export_problem_name,
                    kernel_name=item.stem,
                    json_path=str(item.resolve()),
                    cu_path=str(cu_path.resolve()),
                    reference_path=str(reference_path.resolve()),
                    reference_relpath=reference_relpath,
                    generator_names=generator_names,
                    initially_correct=bool(metadata.get("Correct")),
                    initial_max_diff=normalize_numeric_diff(metadata.get("Max_Diff")),
                    initial_error=metadata.get("Error"),
                    init_arg_names=init_arg_names,
                    forward_arg_names=forward_arg_names,
                    binding_name=binding_name,
                    binding_params=binding_params,
                )
            )

    stats = DiscoveryStats(
        total_scanned=total_scanned,
        eligible=len(candidates),
        skipped_missing_cu=skipped_missing_cu,
        skipped_status_filtered=skipped_status_filtered,
        skipped_unparseable_cuda=skipped_unparseable_cuda,
        ignored_non_json=ignored_non_json,
    )
    return candidates, stats


def make_tasks(
    candidates: list[CandidateSpec],
    gpu_ids: list[int],
    *,
    workers_per_gpu: int,
    cache_root: Path,
    num_correct_trials: int,
    backend: str,
    precision: str,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    capacity = max(1, len(gpu_ids) * max(1, workers_per_gpu))
    next_slot = 0

    for candidate in candidates:
        generator_names = [BASE_GENERATOR_NAME, *candidate.generator_names]
        for generator_name in generator_names:
            gpu_id = gpu_ids[next_slot % capacity % len(gpu_ids)]
            next_slot += 1
            task_key = f"{candidate.solution_key}::{generator_name}"
            build_dir = cache_root / f"gpu{gpu_id}" / str(candidate.task_id) / candidate.kernel_name / generator_name
            tasks.append(
                EvalTask(
                    task_key=task_key,
                    solution_key=candidate.solution_key,
                    generator_name=generator_name,
                    gpu_id=gpu_id,
                    build_dir=str(build_dir),
                    candidate=candidate,
                    num_correct_trials=num_correct_trials,
                    backend=backend,
                    precision=precision,
                )
            )
    return tasks


def evaluate_task(task: EvalTask) -> dict[str, Any]:
    import torch
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

    task_start = time.time()
    ensure_executable_dir_on_path()
    os.makedirs(task.build_dir, exist_ok=True)

    reference_src = build_reference_source(Path(task.candidate.reference_path), task.generator_name)
    custom_src = build_custom_wrapper_source(task)
    device = torch.device(f"cuda:{task.gpu_id}")

    try:
        result = eval_kernel_against_ref(
            original_model_src=reference_src,
            custom_model_src=custom_src,
            measure_performance=False,
            verbose=False,
            num_correct_trials=task.num_correct_trials,
            num_perf_trials=1,
            build_dir=task.build_dir,
            device=device,
            backend=task.backend,
            precision=get_torch_dtype_from_string(task.precision),
        )
        if result is None:
            raise RuntimeError("eval_kernel_against_ref returned None; likely transient compilation failure.")
    except Exception as exc:
        return {
            "task_key": task.task_key,
            "solution_key": task.solution_key,
            "generator_name": task.generator_name,
            "gpu_id": task.gpu_id,
            "build_dir": task.build_dir,
            "task_status": TASK_STATUS_ERROR,
            "correctness": False,
            "compiled": False,
            "duration_sec": round(time.time() - task_start, 6),
            "result": None,
            "metadata": {
                "exception_type": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            },
            "candidate": sanitize_for_json(asdict(task.candidate)),
        }

    return {
        "task_key": task.task_key,
        "solution_key": task.solution_key,
        "generator_name": task.generator_name,
        "gpu_id": task.gpu_id,
        "build_dir": task.build_dir,
        "task_status": TASK_STATUS_PASSED if result.correctness else TASK_STATUS_FAILED,
        "correctness": bool(result.correctness),
        "compiled": bool(result.compiled),
        "duration_sec": round(time.time() - task_start, 6),
        "result": sanitize_for_json(result.dict()),
        "metadata": sanitize_for_json(result.metadata),
        "candidate": sanitize_for_json(asdict(task.candidate)),
    }


def _worker_entry(task: EvalTask, result_queue: mp.Queue):
    try:
        ensure_executable_dir_on_path()
        result_queue.put(evaluate_task(task))
    except Exception as exc:
        result_queue.put(
            {
                "task_key": task.task_key,
                "solution_key": task.solution_key,
                "generator_name": task.generator_name,
                "gpu_id": task.gpu_id,
                "build_dir": task.build_dir,
                "task_status": TASK_STATUS_ERROR,
                "correctness": False,
                "compiled": False,
                "duration_sec": None,
                "result": None,
                "metadata": {
                    "exception_type": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "candidate": sanitize_for_json(asdict(task.candidate)),
            }
        )


def start_task_process(ctx: mp.context.BaseContext, task: EvalTask) -> tuple[mp.Process, mp.Queue, float]:
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_worker_entry, args=(task, result_queue))
    process.start()
    return process, result_queue, time.monotonic()


def load_existing_events(events_path: Path) -> dict[str, dict[str, Any]]:
    if not events_path.exists():
        return {}

    events: dict[str, dict[str, Any]] = {}
    with open(events_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            task_key = event.get("task_key")
            if task_key:
                events[task_key] = event
    return events


def append_event(events_path: Path, event: dict[str, Any]):
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_for_json(event), sort_keys=False))
        handle.write("\n")


def classify_solution_status(base_passed: bool, failing_advanced: list[str], has_advanced: bool) -> str:
    if not has_advanced:
        return FINAL_STATUS_SKIPPED_NO_ADVANCED
    if base_passed and not failing_advanced:
        return FINAL_STATUS_PASSES_ALL
    if not base_passed and not failing_advanced:
        return FINAL_STATUS_FAILS_BASE_ONLY
    if base_passed and failing_advanced:
        return FINAL_STATUS_FAILS_ADVANCED_ONLY
    return FINAL_STATUS_FAILS_BOTH


def build_summary(
    candidates: list[CandidateSpec],
    events: dict[str, dict[str, Any]],
    discovery_stats: DiscoveryStats,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    by_solution: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events.values():
        by_solution.setdefault(event["solution_key"], {})[event["generator_name"]] = event

    solution_entries: list[dict[str, Any]] = []
    totals = {
        "total_scanned": discovery_stats.total_scanned,
        "eligible_candidates": discovery_stats.eligible,
        "skipped_missing_cu": discovery_stats.skipped_missing_cu,
        "skipped_status_filtered": discovery_stats.skipped_status_filtered,
        "skipped_unparseable_cuda": discovery_stats.skipped_unparseable_cuda,
        "completed_task_events": len(events),
        "base_tasks_completed": 0,
        "advanced_tasks_completed": 0,
        FINAL_STATUS_PASSES_ALL: 0,
        FINAL_STATUS_FAILS_BASE_ONLY: 0,
        FINAL_STATUS_FAILS_ADVANCED_ONLY: 0,
        FINAL_STATUS_FAILS_BOTH: 0,
        FINAL_STATUS_SKIPPED_NO_ADVANCED: 0,
        FINAL_STATUS_INCOMPLETE: 0,
    }

    for candidate in candidates:
        generator_results = by_solution.get(candidate.solution_key, {})
        base_event = generator_results.get(BASE_GENERATOR_NAME)
        advanced_events = [generator_results[name] for name in candidate.generator_names if name in generator_results]
        failing_advanced = sorted(event["generator_name"] for event in advanced_events if not event.get("correctness", False))
        expected_task_count = 1 + len(candidate.generator_names)
        completed_task_count = len(generator_results)
        if completed_task_count < expected_task_count:
            final_status = FINAL_STATUS_INCOMPLETE
            base_passed = bool(base_event and base_event.get("correctness"))
        else:
            base_passed = bool(base_event and base_event.get("correctness"))
            final_status = classify_solution_status(base_passed, failing_advanced, bool(candidate.generator_names))

        if base_event is not None:
            totals["base_tasks_completed"] += 1
        totals["advanced_tasks_completed"] += len(advanced_events)
        totals[final_status] += 1

        solution_entries.append(
            {
                **sanitize_for_json(asdict(candidate)),
                "solution_key": candidate.solution_key,
                "final_status": final_status,
                "base_result": sanitize_for_json(base_event),
                "advanced_results": sanitize_for_json(advanced_events),
                "failing_advanced_generators": failing_advanced,
                "completed_task_count": completed_task_count,
                "expected_task_count": expected_task_count,
            }
        )

    return {
        "config": sanitize_for_json(config),
        "totals": totals,
        "solutions": solution_entries,
    }


def render_mismatch_summary(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    lines = [
        f"Total scanned JSON entries: {totals['total_scanned']}",
        f"Eligible candidates: {totals['eligible_candidates']}",
        f"Skipped missing CUDA file: {totals['skipped_missing_cu']}",
        f"Skipped by status filter: {totals['skipped_status_filtered']}",
        f"Skipped unparseable CUDA export: {totals['skipped_unparseable_cuda']}",
        f"Completed task events: {totals['completed_task_events']}",
        f"Completed base tasks: {totals['base_tasks_completed']}",
        f"Completed advanced tasks: {totals['advanced_tasks_completed']}",
        "",
        f"{FINAL_STATUS_PASSES_ALL}: {totals[FINAL_STATUS_PASSES_ALL]}",
        f"{FINAL_STATUS_FAILS_BASE_ONLY}: {totals[FINAL_STATUS_FAILS_BASE_ONLY]}",
        f"{FINAL_STATUS_FAILS_ADVANCED_ONLY}: {totals[FINAL_STATUS_FAILS_ADVANCED_ONLY]}",
        f"{FINAL_STATUS_FAILS_BOTH}: {totals[FINAL_STATUS_FAILS_BOTH]}",
        f"{FINAL_STATUS_SKIPPED_NO_ADVANCED}: {totals[FINAL_STATUS_SKIPPED_NO_ADVANCED]}",
        f"{FINAL_STATUS_INCOMPLETE}: {totals[FINAL_STATUS_INCOMPLETE]}",
        "",
        "Fails advanced only:",
    ]

    advanced_only = [entry for entry in summary["solutions"] if entry["final_status"] == FINAL_STATUS_FAILS_ADVANCED_ONLY]
    if advanced_only:
        for entry in advanced_only:
            lines.append(
                f"- {entry['solution_key']} failed advanced generators: {', '.join(entry['failing_advanced_generators'])}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Fails base only:")
    base_only = [entry for entry in summary["solutions"] if entry["final_status"] == FINAL_STATUS_FAILS_BASE_ONLY]
    if base_only:
        for entry in base_only:
            lines.append(f"- {entry['solution_key']}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Fails both:")
    both = [entry for entry in summary["solutions"] if entry["final_status"] == FINAL_STATUS_FAILS_BOTH]
    if both:
        for entry in both:
            failed = ", ".join(entry["failing_advanced_generators"]) or "no advanced failures recorded"
            lines.append(f"- {entry['solution_key']} failed base and advanced: {failed}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def write_summary_files(summary: dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / SUMMARY_JSON_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)

    with open(output_dir / SUMMARY_CSV_FILENAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "solution_key",
                "task_id",
                "kernel_name",
                "initially_correct",
                "initial_max_diff",
                "final_status",
                "failing_advanced_generators",
            ],
        )
        writer.writeheader()
        for entry in summary["solutions"]:
            writer.writerow(
                {
                    "solution_key": entry["solution_key"],
                    "task_id": entry["task_id"],
                    "kernel_name": entry["kernel_name"],
                    "initially_correct": entry["initially_correct"],
                    "initial_max_diff": entry["initial_max_diff"],
                    "final_status": entry["final_status"],
                    "failing_advanced_generators": ",".join(entry["failing_advanced_generators"]),
                }
            )

    with open(output_dir / MISMATCH_SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
        handle.write(render_mismatch_summary(summary))


def print_console_summary(summary: dict[str, Any]):
    totals = summary["totals"]
    print(f"Total scanned JSON entries: {totals['total_scanned']}")
    print(f"Eligible candidates: {totals['eligible_candidates']}")
    print(f"Skipped missing CUDA file: {totals['skipped_missing_cu']}")
    print(f"Skipped by status filter: {totals['skipped_status_filtered']}")
    print(f"Skipped unparseable CUDA export: {totals['skipped_unparseable_cuda']}")
    print(f"Completed task events: {totals['completed_task_events']}")
    print(f"Completed base tasks: {totals['base_tasks_completed']}")
    print(f"Completed advanced tasks: {totals['advanced_tasks_completed']}")
    print(f"{FINAL_STATUS_PASSES_ALL}: {totals[FINAL_STATUS_PASSES_ALL]}")
    print(f"{FINAL_STATUS_FAILS_BASE_ONLY}: {totals[FINAL_STATUS_FAILS_BASE_ONLY]}")
    print(f"{FINAL_STATUS_FAILS_ADVANCED_ONLY}: {totals[FINAL_STATUS_FAILS_ADVANCED_ONLY]}")
    print(f"{FINAL_STATUS_FAILS_BOTH}: {totals[FINAL_STATUS_FAILS_BOTH]}")
    print(f"{FINAL_STATUS_SKIPPED_NO_ADVANCED}: {totals[FINAL_STATUS_SKIPPED_NO_ADVANCED]}")
    print(f"{FINAL_STATUS_INCOMPLETE}: {totals[FINAL_STATUS_INCOMPLETE]}")


def print_streaming_result(event: dict[str, Any], print_every_result: bool):
    generator_name = event["generator_name"]
    solution_key = event["solution_key"]
    correctness = "pass" if event.get("correctness") else "fail"

    should_print = print_every_result or not event.get("correctness")
    if not should_print:
        return

    print(
        f"[{event['task_status']}] {solution_key} generator={generator_name} gpu={event['gpu_id']} correctness={correctness}",
        flush=True,
    )


def save_current_summary(
    output_dir: Path,
    candidates: list[CandidateSpec],
    events: dict[str, dict[str, Any]],
    discovery_stats: DiscoveryStats,
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = build_summary(candidates, events, discovery_stats, config=config)
    write_summary_files(summary, output_dir)
    return summary


def _signal_handler(signum: int, _frame):
    global _STOP_REQUESTED

    _STOP_REQUESTED = True
    signame = signal.Signals(signum).name
    print(f"\nReceived {signame}; stopping after preserving completed results.", flush=True)


def install_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def execute_tasks(
    tasks: list[EvalTask],
    *,
    timeout: int,
    workers_per_gpu: int,
    output_dir: Path,
    candidates: list[CandidateSpec],
    discovery_stats: DiscoveryStats,
    config: dict[str, Any],
    fail_fast: bool,
    print_every_result: bool,
) -> dict[str, dict[str, Any]]:
    global _STOP_REQUESTED

    events_path = output_dir / EVENTS_FILENAME
    completed_events = load_existing_events(events_path)
    pending = [task for task in tasks if task.task_key not in completed_events]

    if not pending:
        return completed_events

    ctx = mp.get_context("spawn")
    capacity = max(1, len({task.gpu_id for task in tasks}) * max(1, workers_per_gpu))
    active: list[tuple[EvalTask, mp.Process, mp.Queue, float]] = []
    progress = tqdm(total=len(tasks), initial=len(completed_events), desc="Retesting exported kernels", unit="task")

    try:
        while (pending or active) and not _STOP_REQUESTED:
            while pending and len(active) < capacity and not _STOP_REQUESTED:
                task = pending.pop(0)
                process, result_queue, start_time = start_task_process(ctx, task)
                active.append((task, process, result_queue, start_time))

            completed_indexes: list[int] = []
            for index, (task, process, result_queue, start_time) in enumerate(active):
                elapsed = time.monotonic() - start_time
                if process.is_alive() and elapsed > timeout:
                    process.terminate()
                    process.join(5)
                    event = {
                        "task_key": task.task_key,
                        "solution_key": task.solution_key,
                        "generator_name": task.generator_name,
                        "gpu_id": task.gpu_id,
                        "build_dir": task.build_dir,
                        "task_status": TASK_STATUS_TIMEOUT,
                        "correctness": False,
                        "compiled": False,
                        "duration_sec": round(timeout, 6),
                        "result": None,
                        "metadata": {"timeout_seconds": timeout},
                        "candidate": sanitize_for_json(asdict(task.candidate)),
                    }
                elif process.is_alive():
                    continue
                else:
                    process.join(1)
                    try:
                        event = result_queue.get_nowait()
                    except queue.Empty:
                        event = {
                            "task_key": task.task_key,
                            "solution_key": task.solution_key,
                            "generator_name": task.generator_name,
                            "gpu_id": task.gpu_id,
                            "build_dir": task.build_dir,
                            "task_status": TASK_STATUS_ERROR,
                            "correctness": False,
                            "compiled": False,
                            "duration_sec": round(elapsed, 6),
                            "result": None,
                            "metadata": {
                                "exitcode": process.exitcode,
                                "exception": "Worker exited without returning a result",
                            },
                            "candidate": sanitize_for_json(asdict(task.candidate)),
                        }

                completed_events[event["task_key"]] = event
                append_event(events_path, event)
                print_streaming_result(event, print_every_result=print_every_result)
                save_current_summary(output_dir, candidates, completed_events, discovery_stats, config)
                progress.update(1)
                completed_indexes.append(index)

                if fail_fast and not event.get("correctness", False):
                    _STOP_REQUESTED = True

                result_queue.close()
                result_queue.join_thread()

            for index in reversed(completed_indexes):
                del active[index]

            if not completed_indexes and active:
                time.sleep(0.2)
    finally:
        progress.close()
        for _task, process, result_queue, _start_time in active:
            if process.is_alive():
                process.terminate()
                process.join(5)
            result_queue.close()
            result_queue.join_thread()

    return completed_events


def choose_output_dir(output_dir: str | None, resume: str | None) -> Path:
    if resume:
        return Path(resume).resolve()
    if output_dir:
        return Path(output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_RESULTS_ROOT / timestamp).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retest exported KernelBench CUDA kernels on base and advanced generators."
    )
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--problems", default=",".join(str(problem_id) for problem_id in DEFAULT_PROBLEM_IDS))
    parser.add_argument("--gpus")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--print-every-result", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    install_signal_handlers()

    export_root = Path(args.export_root).resolve()
    output_dir = choose_output_dir(args.output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_ids = parse_problem_ids(args.problems)
    gpu_ids = parse_gpu_ids(args.gpus)
    candidates, discovery_stats = discover_candidates(export_root, problem_ids)
    tasks = make_tasks(
        candidates,
        gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        cache_root=output_dir / "build",
        num_correct_trials=args.num_correct_trials,
        backend=args.backend,
        precision=args.precision,
    )

    config = {
        "export_root": str(export_root),
        "problem_ids": list(problem_ids),
        "gpu_ids": gpu_ids,
        "workers_per_gpu": args.workers_per_gpu,
        "timeout": args.timeout,
        "num_correct_trials": args.num_correct_trials,
        "precision": args.precision,
        "backend": args.backend,
        "output_dir": str(output_dir),
        "resume": args.resume,
        "fail_fast": args.fail_fast,
        "print_every_result": args.print_every_result,
    }

    completed_events = execute_tasks(
        tasks,
        timeout=args.timeout,
        workers_per_gpu=args.workers_per_gpu,
        output_dir=output_dir,
        candidates=candidates,
        discovery_stats=discovery_stats,
        config=config,
        fail_fast=args.fail_fast,
        print_every_result=args.print_every_result,
    )

    summary = save_current_summary(output_dir, candidates, completed_events, discovery_stats, config)
    print_console_summary(summary)

    if _STOP_REQUESTED:
        print("Run interrupted. Resume with:", flush=True)
        print(f"  python {Path(__file__).name} --resume {output_dir}", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
