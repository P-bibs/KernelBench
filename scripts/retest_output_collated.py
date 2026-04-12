#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
KERNELBENCH_ROOT = SCRIPT_DIR.parent
REPO_ROOT = KERNELBENCH_ROOT.parent
SRC_DIR = KERNELBENCH_ROOT / "src"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advanced_input_generators import PROBLEM_TO_ADVANCED_INPUTS
from kernelbench.eval import KernelExecResult, eval_kernel_against_ref, get_torch_dtype_from_string


FINAL_STATUS_STILL_CORRECT = "still_correct"
FINAL_STATUS_BUG_REVEALED = "bug_revealed"
FINAL_STATUS_SKIPPED = "skipped_no_advanced_generator"

TASK_STATUS_PASSED = "passed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_TIMEOUT = "timeout"
TASK_STATUS_ERROR = "error"


def ensure_executable_dir_on_path():
    executable_dir = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if executable_dir not in path_entries:
        os.environ["PATH"] = executable_dir if not current_path else f"{executable_dir}{os.pathsep}{current_path}"


ensure_executable_dir_on_path()


@dataclass(frozen=True)
class SolutionSpec:
    level: str
    problem_id: int
    solution_id: int
    solution_path: str
    reference_path: str
    reference_relpath: str
    generator_names: tuple[str, ...]

    @property
    def solution_key(self) -> str:
        return f"{self.level}/{self.problem_id}/{self.solution_id}"


@dataclass(frozen=True)
class EvalTask:
    solution_key: str
    solution_path: str
    reference_path: str
    reference_relpath: str
    generator_name: str
    gpu_id: int
    build_dir: str
    num_correct_trials: int
    backend: str
    precision: str


def normalize_level(level: str | None) -> str | None:
    if level is None:
        return None
    level = level.strip()
    if not level:
        return None
    if level.startswith("level"):
        suffix = level[5:]
    else:
        suffix = level
    if not suffix.isdigit():
        raise ValueError(f"Invalid level: {level}")
    return f"level{int(suffix)}"


def parse_gpu_ids(gpus: str | None) -> list[int]:
    if gpus is None:
        count = torch.cuda.device_count()
        if count <= 0:
            raise ValueError("No visible CUDA devices found. Pass --gpus explicitly or run on a CUDA host.")
        return list(range(count))

    gpu_ids: list[int] = []
    for item in gpus.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("GPU list is empty after parsing --gpus.")
    return gpu_ids


def discover_correct_solutions(
    collated_root: Path,
    kernelbench_root: Path,
    *,
    level: str | None = None,
    problem_ids: set[int] | None = None,
) -> list[SolutionSpec]:
    normalized_level = normalize_level(level)
    specs: list[SolutionSpec] = []

    for solution_path in sorted(collated_root.glob("level*/[0-9]*/correct/*.py"), key=_solution_sort_key):
        path_level = solution_path.parts[-4]
        path_problem_id = int(solution_path.parts[-3])

        if normalized_level is not None and path_level != normalized_level:
            continue
        if problem_ids and path_problem_id not in problem_ids:
            continue

        reference_path = resolve_reference_problem(solution_path, kernelbench_root)
        reference_relpath = reference_path.relative_to(kernelbench_root).as_posix()
        generator_names = tuple(sorted(PROBLEM_TO_ADVANCED_INPUTS.get(reference_relpath, frozenset())))

        specs.append(
            SolutionSpec(
                level=path_level,
                problem_id=path_problem_id,
                solution_id=int(solution_path.stem),
                solution_path=str(solution_path.resolve()),
                reference_path=str(reference_path.resolve()),
                reference_relpath=reference_relpath,
                generator_names=generator_names,
            )
        )
    return specs


def resolve_reference_problem(solution_path: Path, kernelbench_root: Path) -> Path:
    level = solution_path.parts[-4]
    problem_id = int(solution_path.parts[-3])
    candidate_dir = kernelbench_root / level
    matches = sorted(candidate_dir.glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one KernelBench problem match for {solution_path}, found {len(matches)} in {candidate_dir}"
        )
    return matches[0]


def create_wrapper_reference_source(problem_path: Path, generator_name: str) -> str:
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


def make_eval_tasks(
    solutions: list[SolutionSpec],
    gpu_ids: list[int],
    *,
    cache_root: Path,
    num_correct_trials: int,
    backend: str,
    precision: str,
    max_tasks: int | None,
) -> tuple[list[EvalTask], list[SolutionSpec]]:
    tasks: list[EvalTask] = []
    included_solutions: list[SolutionSpec] = []
    next_gpu_index = 0

    for solution in solutions:
        if not solution.generator_names:
            included_solutions.append(solution)
            continue

        if max_tasks is not None and tasks and len(tasks) + len(solution.generator_names) > max_tasks:
            break

        if max_tasks is not None and not tasks and len(solution.generator_names) > max_tasks:
            break

        included_solutions.append(solution)
        for generator_name in solution.generator_names:
            gpu_id = gpu_ids[next_gpu_index % len(gpu_ids)]
            next_gpu_index += 1
            build_dir = cache_root / solution.level / str(solution.problem_id) / str(solution.solution_id) / generator_name
            tasks.append(
                EvalTask(
                    solution_key=solution.solution_key,
                    solution_path=solution.solution_path,
                    reference_path=solution.reference_path,
                    reference_relpath=solution.reference_relpath,
                    generator_name=generator_name,
                    gpu_id=gpu_id,
                    build_dir=str(build_dir),
                    num_correct_trials=num_correct_trials,
                    backend=backend,
                    precision=precision,
                )
            )
    return tasks, included_solutions


def evaluate_task(task: EvalTask) -> dict[str, Any]:
    task_start = time.time()
    ensure_executable_dir_on_path()
    os.makedirs(task.build_dir, exist_ok=True)

    with open(task.solution_path, "r", encoding="utf-8") as f:
        kernel_src = f.read()

    wrapper_src = create_wrapper_reference_source(Path(task.reference_path), task.generator_name)
    device = torch.device(f"cuda:{task.gpu_id}")

    try:
        result = eval_kernel_against_ref(
            original_model_src=wrapper_src,
            custom_model_src=kernel_src,
            measure_performance=False,
            verbose=False,
            num_correct_trials=task.num_correct_trials,
            num_perf_trials=1,
            build_dir=task.build_dir,
            device=device,
            backend=task.backend,
            precision=get_torch_dtype_from_string(task.precision),
        )
    except Exception as exc:
        return {
            "solution_key": task.solution_key,
            "solution_path": task.solution_path,
            "reference_relpath": task.reference_relpath,
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
        }

    return {
        "solution_key": task.solution_key,
        "solution_path": task.solution_path,
        "reference_relpath": task.reference_relpath,
        "generator_name": task.generator_name,
        "gpu_id": task.gpu_id,
        "build_dir": task.build_dir,
        "task_status": TASK_STATUS_PASSED if result.correctness else TASK_STATUS_FAILED,
        "correctness": bool(result.correctness),
        "compiled": bool(result.compiled),
        "duration_sec": round(time.time() - task_start, 6),
        "result": sanitize_for_json(result.dict()),
        "metadata": sanitize_for_json(result.metadata),
    }


def _worker_entry(task: EvalTask, result_queue: mp.Queue):
    try:
        ensure_executable_dir_on_path()
        result_queue.put(evaluate_task(task))
    except Exception as exc:
        result_queue.put(
            {
                "solution_key": task.solution_key,
                "solution_path": task.solution_path,
                "reference_relpath": task.reference_relpath,
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
            }
        )


def run_task_batch(tasks: list[EvalTask], timeout: int) -> list[dict[str, Any]]:
    if not tasks:
        return []

    ctx = mp.get_context("spawn")
    processes: list[tuple[EvalTask, mp.Process, mp.Queue, float]] = []
    results: list[dict[str, Any]] = []

    for task in tasks:
        result_queue: mp.Queue = ctx.Queue(maxsize=1)
        process = ctx.Process(target=_worker_entry, args=(task, result_queue))
        process.start()
        processes.append((task, process, result_queue, time.monotonic()))

    for task, process, result_queue, start_time in processes:
        remaining = timeout - (time.monotonic() - start_time)
        if remaining > 0:
            process.join(remaining)
        if process.is_alive():
            process.terminate()
            process.join(5)
            results.append(
                {
                    "solution_key": task.solution_key,
                    "solution_path": task.solution_path,
                    "reference_relpath": task.reference_relpath,
                    "generator_name": task.generator_name,
                    "gpu_id": task.gpu_id,
                    "build_dir": task.build_dir,
                    "task_status": TASK_STATUS_TIMEOUT,
                    "correctness": False,
                    "compiled": False,
                    "duration_sec": round(timeout, 6),
                    "result": None,
                    "metadata": {"timeout_seconds": timeout},
                }
            )
        else:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                results.append(
                    {
                        "solution_key": task.solution_key,
                        "solution_path": task.solution_path,
                        "reference_relpath": task.reference_relpath,
                        "generator_name": task.generator_name,
                        "gpu_id": task.gpu_id,
                        "build_dir": task.build_dir,
                        "task_status": TASK_STATUS_ERROR,
                        "correctness": False,
                        "compiled": False,
                        "duration_sec": None,
                        "result": None,
                        "metadata": {"exitcode": process.exitcode, "exception": "Worker exited without returning a result"},
                    }
                )
        result_queue.close()
        result_queue.join_thread()

    return results


def execute_tasks_in_parallel(
    tasks: list[EvalTask],
    gpu_ids: list[int],
    *,
    timeout: int,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    batch_size = max(1, len(gpu_ids))

    remaining = list(tasks)
    while remaining:
        current_batch = remaining[:batch_size]
        remaining = remaining[batch_size:]
        batch_results = run_task_batch(current_batch, timeout=timeout)
        results.extend(batch_results)
        if fail_fast and any(not result["correctness"] for result in batch_results):
            break
    return results


def build_summary(
    solutions: list[SolutionSpec],
    task_results: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    by_solution: dict[str, list[dict[str, Any]]] = {}
    for result in task_results:
        by_solution.setdefault(result["solution_key"], []).append(result)

    solution_entries: list[dict[str, Any]] = []
    still_correct = 0
    bug_revealed = 0
    skipped = 0

    for solution in solutions:
        task_entries = sorted(by_solution.get(solution.solution_key, []), key=lambda item: item["generator_name"])
        if not solution.generator_names:
            final_status = FINAL_STATUS_SKIPPED
            failing_generators: list[str] = []
            skipped += 1
        else:
            failing_generators = [entry["generator_name"] for entry in task_entries if not entry["correctness"]]
            if failing_generators:
                final_status = FINAL_STATUS_BUG_REVEALED
                bug_revealed += 1
            else:
                final_status = FINAL_STATUS_STILL_CORRECT
                still_correct += 1

        solution_entries.append(
            {
                **asdict(solution),
                "solution_key": solution.solution_key,
                "final_status": final_status,
                "failing_generators": failing_generators,
                "generator_results": task_entries,
            }
        )

    totals = {
        "total_scanned": len(solutions),
        "covered_solutions": sum(1 for solution in solutions if solution.generator_names),
        "skipped_no_advanced_generator": skipped,
        "still_correct": still_correct,
        "bug_revealed": bug_revealed,
        "total_generator_tasks": len(task_results),
    }

    return {
        "config": sanitize_for_json(config),
        "totals": totals,
        "solutions": solution_entries,
    }


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


def write_summary_files(summary: dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solution_key",
                "solution_path",
                "reference_relpath",
                "final_status",
                "generator_names",
                "failing_generators",
            ],
        )
        writer.writeheader()
        for solution in summary["solutions"]:
            writer.writerow(
                {
                    "solution_key": solution["solution_key"],
                    "solution_path": solution["solution_path"],
                    "reference_relpath": solution["reference_relpath"],
                    "final_status": solution["final_status"],
                    "generator_names": ",".join(solution["generator_names"]),
                    "failing_generators": ",".join(solution["failing_generators"]),
                }
            )


def print_console_summary(summary: dict[str, Any]):
    totals = summary["totals"]
    print(f"Scanned correct solutions: {totals['total_scanned']}")
    print(f"Covered by advanced generators: {totals['covered_solutions']}")
    print(f"Skipped (no advanced generator): {totals['skipped_no_advanced_generator']}")
    print(f"Still correct: {totals['still_correct']}")
    print(f"Bug revealed: {totals['bug_revealed']}")
    print(f"Generator tasks executed: {totals['total_generator_tasks']}")

    bug_entries = [solution for solution in summary["solutions"] if solution["final_status"] == FINAL_STATUS_BUG_REVEALED]
    skipped_entries = [solution for solution in summary["solutions"] if solution["final_status"] == FINAL_STATUS_SKIPPED]

    if bug_entries:
        print("\nRevealed bugs:")
        for entry in bug_entries:
            print(f"- {entry['solution_key']} ({Path(entry['solution_path']).name}) failed: {', '.join(entry['failing_generators'])}")

    if skipped_entries:
        print("\nSkipped (no advanced generator):")
        for entry in skipped_entries:
            print(f"- {entry['solution_key']} ({entry['reference_relpath']})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retest collated KernelBench solutions with advanced input generators.")
    parser.add_argument("--collated-root", default=str(REPO_ROOT / "output_collated"))
    parser.add_argument("--kernelbench-root", default=str(REPO_ROOT / "KernelBench" / "KernelBench"))
    parser.add_argument("--gpus", default=None, help="Comma-separated CUDA device ids. Defaults to all visible GPUs.")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--level", default=None, help="Filter to a single level, e.g. level1 or 1.")
    parser.add_argument("--problem-id", action="append", type=int, default=None, help="Filter to one or more problem ids.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Cap the number of generator tasks by including only whole solutions whose generators fit within the limit.",
    )
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "logs" / "output_collated_retest" / timestamp


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    collated_root = Path(args.collated_root).resolve()
    kernelbench_root = Path(args.kernelbench_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir()
    cache_root = REPO_ROOT / "cache" / "output_collated_retest"
    gpu_ids = parse_gpu_ids(args.gpus)

    problem_ids = set(args.problem_id) if args.problem_id else None
    solutions = discover_correct_solutions(
        collated_root,
        kernelbench_root,
        level=args.level,
        problem_ids=problem_ids,
    )
    tasks, included_solutions = make_eval_tasks(
        solutions,
        gpu_ids,
        cache_root=cache_root,
        num_correct_trials=args.num_correct_trials,
        backend=args.backend,
        precision=args.precision,
        max_tasks=args.max_tasks,
    )

    task_results = execute_tasks_in_parallel(
        tasks,
        gpu_ids,
        timeout=args.timeout,
        fail_fast=args.fail_fast,
    )
    if args.fail_fast and len(task_results) < len(tasks):
        completed_solution_keys = {result["solution_key"] for result in task_results}
        included_solutions = [
            solution
            for solution in included_solutions
            if not solution.generator_names or solution.solution_key in completed_solution_keys
        ]

    config = {
        "collated_root": str(collated_root),
        "kernelbench_root": str(kernelbench_root),
        "gpus": gpu_ids,
        "num_correct_trials": args.num_correct_trials,
        "timeout": args.timeout,
        "precision": args.precision,
        "backend": args.backend,
        "output_dir": str(output_dir),
        "level": normalize_level(args.level),
        "problem_ids": sorted(problem_ids) if problem_ids else None,
        "fail_fast": args.fail_fast,
        "max_tasks": args.max_tasks,
        "evaluated_solution_count": len(included_solutions),
    }
    summary = build_summary(included_solutions, task_results, config=config)
    write_summary_files(summary, output_dir)
    print_console_summary(summary)
    print(f"\nSummary written to: {output_dir}")
    return 0


def _solution_sort_key(path: Path) -> tuple[int, int, int]:
    level = int(path.parts[-4].removeprefix("level"))
    problem_id = int(path.parts[-3])
    solution_id = int(path.stem)
    return (level, problem_id, solution_id)


if __name__ == "__main__":
    raise SystemExit(main())
