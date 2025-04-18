import ast
from datasets import load_dataset
from question.predict import PredictionQuestion
from loader.base import QuestionDataset


class Crux(QuestionDataset):
    def __init__(self, path: str = "cruxeval-org/cruxeval", **kwargs) -> None:
        self.path = path
        self.raw = self.load(**kwargs)
        self.questions_list = self.load_questions()
        
    @classmethod
    def load_file(self, file_path: str) -> "Crux":
        return super().load_file(file_path, PredictionQuestion)
    
    @classmethod
    def load_perturb(cls, perturb_tag, task: str = "output") -> "Crux":
        if task not in ["output", "input"]:
            raise ValueError(f"Invalid task: {task}. Must be 'output' or 'input'.")
        ins = cls()
        if perturb_tag == "VAN":
            return ins
        else:
            path = "CUHK-ARISE/CodeCrash"
            data = load_dataset(path, data_files=f"crux_{perturb_tag}_{task}.jsonl")
            ins.questions_list = [PredictionQuestion.from_dict(entry) for entry in data["train"]]
        return ins

    def load(self, **kwargs) -> list[dict]:
        data = load_dataset(self.path)
        return data["test"]

    def load_questions(self) -> list[PredictionQuestion]:
        self.key = "id"
        questions_list = []
        for entry in self.raw:
            code = ast.unparse(ast.parse(entry["code"]))
            questions_list.append(PredictionQuestion(
                dataset="cruxeval",
                function_name="f",
                task_id=entry["id"],
                code=code,
                function_call=f"f({entry['input']})",
                output=entry["output"],
                output_format={**entry}
            ))
        return questions_list

    def load_saved_execution(self, file_path: str, n: int) -> dict:
        saved_execution = super().load_saved_execution(file_path)
        target_count = {qs.task_id: n for qs in self.questions_list}
        
        if saved_execution is None:
            return target_count
        
        task_count = {qid: len(answers) for qid, answers in saved_execution.items()}

        rest_count = {
            tid: target_count[tid] - task_count.get(tid, 0)
            for tid in target_count
        }
        
        return rest_count