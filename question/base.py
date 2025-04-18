import copy
from typing import List

from llm import LLMChat, Message
from question.solution import Solution
from utils import *


class BaseQuestion:
    def __init__(self, task_id: str, dataset: str) -> None:
        self.task_id = task_id
        self.dataset = dataset

    def to_dict(self) -> dict:
        state = self.__dict__.copy()
        return state

    def request(self, model: LLMChat, inputs: List[Message], **kwargs) -> List[str]:
        n = kwargs.get("n", 1)
        answers = []
        while len(answers) < n:
            responses = model.chat(messages=inputs, n=n-len(answers))
            answers += responses
        return answers

    def run(self, model: LLMChat, inputs: List[Message], extraction_function, **kwargs) -> str:
        n = kwargs.get("n", 1)
        key = kwargs.get("key")

        answers = self.request(model, inputs, n=n)
        return Solution(copy.deepcopy(self), answers)
