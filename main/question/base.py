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
        usage = None
        while len(answers) < n:
            responses, usage = model.chat(messages=inputs, n=n-len(answers))
            answers += responses
        return answers, usage

    def run(self, model: LLMChat, inputs: List[Message], extraction_function, **kwargs) -> str:
        n = kwargs.get("n", 1)
        key = kwargs.get("key")

        answers, usage = self.request(model, inputs, n=n)
        return Solution(copy.deepcopy(self), answers, usage=usage)
