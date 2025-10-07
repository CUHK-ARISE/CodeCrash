from typing import Optional

from perturbations.textual.transformers import MisleadingInsertor, FunctionHintInserter
from utils.format import Task


def process_inserting_misleading_comments(code: str, output: Optional[str] = None, once: bool = False, p: float = 1.0, **kwargs) -> str:
    return MisleadingInsertor(code, mode="comment", output=output, once=once, p=p).run()


def process_inserting_misleading_prints(code: str, output: Optional[str] = None, once: bool = False, p: float = 1.0, **kwargs) -> str:
    return MisleadingInsertor(code, mode="print", output=output, once=once, p=p).run()


def process_inserting_function_hints(code: str, task: Task, target_function: str, incorrect_value: str, **kwargs) -> str:
    if task == Task.INPUT_PREDICTION:
        text = f"The function call is {incorrect_value}"
    elif task == Task.OUTPUT_PREDICTION:
        text = f"The return value is {incorrect_value}"
    else:
        raise ValueError("task must be either 'Task.INPUT_PREDICTION' or 'Task.OUTPUT_PREDICTION'")
    return FunctionHintInserter(code, target_function, text).run(task.value)
