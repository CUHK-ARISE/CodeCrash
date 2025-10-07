import os
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from evaluate.execute import verify_correctness


def evaluate_single_solution(code: str, function_name: str, function_call: str, output: Any, sol: str, timeout: int) -> bool:
    if function_name in sol:
        expr = f"assert {sol} == {output}"
    else:
        expr = f"assert {function_call} == {sol}"
    result = verify_correctness(code=code, test=expr, timeout=timeout)
    return result["status"] == "passed"


def evaluate_solution(
    filepath: str,
    output_file: Optional[str] = None,
    timeout: int = 60,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Evaluate solutions in parallel, solution-level granularity.
    """
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if output_file is None:
        output_file = filepath.replace(".jsonl", "_eval.json")

    correctness_dict = {}
    output_dict = {
        "pass_at_1": 0.0,
        "raw_generations": {},
        "raw_scored_generations": {},
    }

    jobs = []
    for entry in data:
        for sol in entry["solutions"]:
            jobs.append((entry, sol))

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_solution,
                entry["code"],
                entry["function_name"],
                entry["function_call"],
                entry["output"],
                sol,
                timeout,
            ): (entry["task_id"], sol, entry)
            for entry, sol in jobs
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            task_id, sol, entry = futures[future]
            try:
                passed = future.result()
            except Exception:
                passed = False

            if task_id not in results:
                results[task_id] = []
            results[task_id].append((sol, passed))

    for entry in data:
        task_id = entry["task_id"]
        if task_id in results:
            sols, correctness = zip(*results[task_id])
            acc = sum(correctness) / len(correctness)
            correctness_dict[task_id] = (list(correctness), acc)
            output_dict["raw_generations"][task_id] = list(sols)
            output_dict["raw_scored_generations"][task_id] = list(correctness)
        else:
            correctness_dict[task_id] = ([], 0.0)

    mean_score = float(np.mean([v[1] for v in correctness_dict.values()]))
    output_dict["pass_at_1"] = mean_score

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_dict, f)

    print(f"Pass@1: {mean_score*100:.2f}")
    return {"Pass@1": mean_score}