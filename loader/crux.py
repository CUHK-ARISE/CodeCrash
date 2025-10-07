import os
import ast
from datasets import load_dataset
from typing import List, Dict, Any

from question.base import Question
from loader.base import QuestionDataset


class CruxEval(QuestionDataset):
    dataset_name = "crux"
    dataset_path = "cruxeval-org/cruxeval"
    cache_file = "./datasets/crux.jsonl"
    questions_list: List[Question] = []

    def __init__(self):
        """Load CRUXEval dataset (prefer local cache)."""
        if os.path.exists(self.cache_file):
            print("🔍 Loading from local cache...")
            data = load_dataset("json", data_files=self.cache_file, split="train")
        else:
            print("🌐 Downloading from HuggingFace...")
            data = load_dataset(self.dataset_path, split="test")
            data.to_json(self.cache_file)
        self.load_questions(data)

    def load_questions(self, data: List[Dict[str, Any]]) -> None:
        for entry in data:
            task_id = entry["id"]
            code = ast.unparse(ast.parse(entry["code"]))    # Standardize code formatting
            function_name = "f"
            function_call = f"f({entry['input']})"
            output = entry["output"]
            test = f"assert {function_call} == {output}"

            self.questions_list.append(Question(
                task_id=task_id,
                dataset=self.dataset_name,
                code=code,
                function_name=function_name,
                function_call=function_call,
                output=output,
                test=test
            ))
