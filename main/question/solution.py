# Modules
from evaluators import *
from utils.saving import *
from datetime import datetime

def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()

class Solution:
    def __init__(self, task_id: str, dataset: str, input_prompt: str, solutions: list[set[str, str]], output_format: dict, usage: dict = None) -> None:
        self.task_id = task_id
        self.dataset = dataset
        self.input_prompt = input_prompt
        self.solutions = solutions
        self.output_format = output_format
        self.usage = usage

    def save_lcb(self, file_name: str) -> None:
        answer_key = {
            "LCB-Execution": "pred_list"
        }.get(self.dataset)
        output_list, answer_list = zip(*self.solutions)
        file_path = f"{file_name}.json"

        try:
            with open(file_path, "r") as file:
                existing_data = json.load(file)
        except:
            existing_data = []

        entry = next((item for item in existing_data if item.get("id") == self.task_id), None)
        if entry:
            entry["output_list"].extend(output_list)
            entry[answer_key].extend(answer_list)
        else:
            entry = {
                **self.output_format,
                "input_prompt": self.input_prompt,
                "output_list": list(output_list),
                answer_key: list(answer_list),
            }
            if self.usage:
                entry["usages"] = self.usage
            existing_data.append(entry)

        with open(file_path, "w") as file:
            json.dump(existing_data, file, indent=4, default=json_serial)

    def save_crux(self, file_name: str) -> None:
        output_list, answer_list = zip(*self.solutions)
        data_file_path = f"{file_name}.json"
        record_file_path = f"{file_name}_record.json"

        try:
            with open(data_file_path, "r") as file:
                existing_data = json.load(file)
            with open(record_file_path, "r") as file:
                existing_record = json.load(file)
        except:
            existing_data = {}
            existing_record = []

        if self.task_id in existing_data:
            existing_data[self.task_id].extend(answer_list)
        else:
            existing_data[self.task_id] = list(answer_list)

        entry = next((item for item in existing_record if item.get("id") == self.task_id), None)
        if entry:
            entry["output_list"].extend(output_list)
            entry["answer_list"].extend(answer_list)
        else:
            entry = {
                **self.output_format,
                "input_prompt": self.input_prompt,
                "output_list": list(output_list),
                "answer_list": list(answer_list),
            }
            if self.usage:
                entry["usages"] = self.usage
            existing_record.append(entry)
        
        with open(data_file_path, "w") as file:
            json.dump(existing_data, file, indent=4)
        with open(record_file_path, "w") as file:
            json.dump(existing_record, file, indent=4)

    def save(self, file_name: str = "save") -> None:
        if self.dataset.startswith("LCB"):
            self.save_lcb(file_name)
        elif self.dataset in ["cruxeval"]:
            self.save_crux(file_name)
