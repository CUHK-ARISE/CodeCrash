import os
import json
from datetime import datetime
from datasets import load_dataset
from typing import List

from question.base import Question
from utils.format import Task, Perturbation


def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

class QuestionDataset:
    dataset_name: str = "customize"
    dataset_path: str = ""
    questions_list: List[Question] = []
    
    @classmethod
    def load_file(cls, file_path: str) -> "QuestionDataset":
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
        ins = cls.__new__(cls)
        ins.dataset_path = file_path
        QuestionDataset.__init__(ins)
        ins.questions_list = [Question.from_dict(question) for question in data]
        ins.dataset_name = ins.questions_list[0].dataset if ins.questions_list[0].dataset else "customize"
        return ins
    
    @classmethod
    def load_perturb(cls, perturbation: Perturbation, task: Task) -> "QuestionDataset":
        if cls is QuestionDataset:
            raise NotImplementedError("Please call load_perturb() from a subclass (e.g., Crux, LiveCodeBench).")

        ins = cls.__new__(cls)
        QuestionDataset.__init__(ins)

        if perturbation == Perturbation.VAN:
            return cls()

        path = "CUHK-ARISE/CodeCrash"
        try:
            dataset = load_dataset(path, data_files=f"{cls.dataset_name}_{perturbation.value}_{task.value}.jsonl")["train"]
            ins.questions_list = [Question.from_dict(entry) for entry in dataset]
            ins.dataset_name = cls.dataset_name
            return ins
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def save(self, name: str) -> None:
        """ Saves the question list to the specified path in JSONL format. """
        os.makedirs("./customize_datasets", exist_ok=True)
        with open(f"./customize_datasets/{name}.jsonl", "w") as file:
            for question in self.questions_list:
                file.write(json.dumps(question.to_dict(), default=json_serial) + '\n')
