import os
import json
from typing import Dict, Optional
from generate.solutions import Solutions
from loader import QuestionDataset

def save_sol(solutions: Solutions, filepath: str) -> None:  
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
            data = {entry["task_id"]: entry for entry in data}
    
    data[solutions.question.task_id] = solutions.to_dict()

    with open(filepath, "w") as f:
        for entry in data.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_existing_solutions(filepath: str) -> Dict[str, str]:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
            data = {entry["task_id"]: entry for entry in data}
        return data
    return {}
