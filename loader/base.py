import os
import wget
import json
from typing import Optional
from question.base import BaseQuestion
from utils.loading import load_file
from utils.saving import json_serial


class QuestionDataset():
    
    def __init__(self, path: str, **kwargs) -> None:
        self.path = path
        self.raw = self.load(**kwargs)
        self.questions_list = self.load_questions()

    def load(self, **kwargs) -> dict:
        if self.path is None:
            url = kwargs.get("url")
            file_name = kwargs.get("file_name")
            self.path = self.download(url, file_name)

        key = kwargs.get("key", "task_id")
        return load_file(self.path, key)

    def download(self, url: str, file_name: str) -> str:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dataset_loader")
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, file_name)
        if not os.path.exists(file_path):
            wget.download(url, file_path)
            print(f"\nDownloaded {file_name} to: {cache_dir}")
        return file_path

    def load_questions(self) -> list[BaseQuestion]:
        return [BaseQuestion(**qs) for key, qs in self.raw.items()]

    @classmethod
    def load_file(cls, file_path: str, question_type, **kwargs) -> "QuestionDataset":
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
        instance = cls(**kwargs)
        instance.path = file_path
        instance.questions_list = [question_type.from_dict(question) for question in data]
        return instance
    
    def load_saved_execution(self, file_path: str) -> Optional[dict|list]:
        if file_path.endswith(".json"):
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    return json.load(file)
            else:
                return None
        elif file_path.endswith(".jsonl"):
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    return [json.loads(line) for line in file]
            else:
                return None
    
    def save(self, name) -> None:
        """Saves the question list to the specified path in JSONL format."""
        os.makedirs("./dataset_loader/customize", exist_ok=True)
        with open(f"./dataset_loader/customize/{name}.jsonl", "w") as file:
            for question in self.questions_list:
                file.write(json.dumps(question.to_dict(), default=json_serial) + '\n')
