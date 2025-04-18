import ast
from datasets import load_dataset
from question.predict import PredictionQuestion
from loader.base import QuestionDataset


class LiveCodeBench(QuestionDataset):
    def __init__(self, path: str = None, **kwargs) -> None:
        super().__init__(path, **kwargs)

    @classmethod
    def load_file(self, file_path: str) -> "LiveCodeBench":
        return super().load_file(file_path, PredictionQuestion)
    
    @classmethod
    def load_perturb(cls, perturb_tag, task: str = "output") -> "LiveCodeBench":
        if task not in ["output"]:
            raise ValueError(f"Invalid task: {task}. Must be 'output'.")
        ins = cls()
        if perturb_tag == "VAN":
            return ins
        else:
            path = "CUHK-ARISE/CodeCrash"
            data = load_dataset(path, data_files=f"lcb_{perturb_tag}_{task}.jsonl")
            ins.questions_list = [PredictionQuestion.from_dict(entry) for entry in data["train"]]
        return ins

    def load(self, **kwargs) -> list[dict]:
        data = load_dataset("livecodebench/execution-v2")
        return data["test"]

    def load_questions(self) -> list:
        questions_list = []
        for entry in self.raw:
            question = self.load_execution(entry)
            questions_list.append(question)
        return questions_list

    def load_execution(self, entry: dict) -> PredictionQuestion:
        self.key = "id"
        code = ast.unparse(ast.parse(entry["code"]))
        
        return PredictionQuestion(
            task_id=entry["id"],
            dataset="LCB-Execution",
            code=code,
            function_call=entry["input"],
            output=entry["output"],
            function_name=entry["function_name"],
            output_format={
                **entry,
                "input_prompt": "",
                "output_list": [],
                "pred_list": [],
            },
        )
    
    def load_saved_execution(self, file_path: str, n: int) -> dict:
        saved_execution = super().load_saved_execution(file_path)
        target_count = {qs.task_id: n for qs in self.questions_list}
        
        if saved_execution is None:
            return target_count
        
        task_count = {
            entry["id"]: len(entry["pred_list"])
            for entry in saved_execution
        }
        
        rest_count = {
            tid: target_count[tid] - task_count.get(tid, 0)
            for tid in target_count
        }
        
        return rest_count