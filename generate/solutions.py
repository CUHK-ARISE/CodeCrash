import warnings
from typing import List, Dict, Optional

from question.base import Question


class Solutions:
    question: Question
    input_prompt: str
    responses: List[str]
    solutions: List[Optional[str]]

    def __init__(self, question: Question, input_prompt: str = "", responses: List[str] = [], solutions: List[Optional[str]] = []) -> None:
        self.question = question
        self.input_prompt = input_prompt
        self.responses = responses
        self.solutions = solutions
    
    @classmethod
    def merge(cls, sol1: "Solutions", sol2: "Solutions") -> "Solutions":
        if sol1.question.task_id != sol2.question.task_id or sol1.question.dataset != sol2.question.dataset:
            raise ValueError("Different Questions: Cannot merge solutions for different questions.")
        
        if sol1.question.code != sol2.question.code:
            raise ValueError("Perturbation Type Different: Cannot merge solutions for different code.")
        
        input_prompt = sol1.input_prompt

        if sol1.input_prompt != sol2.input_prompt:
            if sol1.input_prompt == "" or sol2.input_prompt == "":
                input_prompt = sol1.input_prompt if sol1.input_prompt != "" else sol2.input_prompt
            else:
                warnings.warn(f"[Error on question {sol1.question.task_id}]: Different Input Prompts: Merging solutions with different input prompts.")
        
        return cls(
            question=sol1.question,
            input_prompt=input_prompt,
            responses=sol1.responses + sol2.responses,
            solutions=sol1.solutions + sol2.solutions,
        )
    
    def add_response(self, response: str, solution: str) -> None:
        self.responses.append(response)
        self.solutions.append(solution)
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.question.task_id,
            "dataset": self.question.dataset,
            "code": self.question.code,
            "function_name": self.question.function_name,
            "function_call": self.question.function_call,
            "output": self.question.output,
            "test": self.question.test,
            "input_prompt": self.input_prompt,
            "responses": self.responses,
            "solutions": self.solutions,
        }
