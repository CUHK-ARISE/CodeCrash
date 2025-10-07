from utils.format import Task, Mode, Perturbation
from evaluate import verify_correctness, verify_correctness_unsafe
from prompt.prompts import get_incorrect_gen_prompt
from prompt.gen_prompts import get_run_prompt
from perturbations.textual import process_inserting_misleading_comments, process_inserting_misleading_prints, process_inserting_function_hints
from perturbations.structural import process_renaming_entities, process_reformatting_conditions, process_inserting_garbage_code, process_psc_all

class Question:
    task_id: str
    dataset: str
    code: str
    function_name: str
    function_call: str
    output: str
    test: str
    
    def __init__(
        self,
        task_id: str,
        dataset: str,
        code: str,
        function_name: str,
        function_call: str,
        output: str,
        test: str = None,
        **kwargs
    ) -> None:
        self.task_id = task_id
        self.dataset = dataset
        self.code = code
        self.function_name = function_name
        self.function_call = function_call
        self.output = output
        self.test = test if test is not None else ""
    
    def to_dict(self) -> dict:
        state = self.__dict__.copy()
        return state
    
    @classmethod
    def from_dict(cls, state: dict) -> "Question":
        return cls(**state)
    
    def perturbate(self, tag: Perturbation, safe_eval: bool, **kwargs) -> None:
        new_code, new_function_name, new_function_call, new_test = self.code, self.function_name, self.function_call, self.test
        if tag == Perturbation.REN:
            new_code, new_function_name, [new_function_call, new_test] = process_renaming_entities(self.code, self.function_name, [self.function_call, self.test])
        elif tag == Perturbation.RTF:
            new_code = process_reformatting_conditions(self.code)
        elif tag == Perturbation.GBC:
            new_code = process_inserting_garbage_code(self.code)
        elif tag == Perturbation.PSC_ALL:
            new_code, new_function_name, [new_function_call, new_test] = process_psc_all(self.code, self.function_name, [self.function_call, self.test])
        elif tag == Perturbation.MCC:
            new_code = process_inserting_misleading_comments(self.code, output=self.output, **kwargs)
        elif tag == Perturbation.MPS:
            new_code = process_inserting_misleading_prints(self.code, output=self.output, **kwargs)
        elif tag == Perturbation.MHC:
            task = kwargs.pop("task")
            new_code = process_inserting_function_hints(self.code, task=task, target_function=self.function_name, **kwargs)
        else:
            raise ValueError(f"Invalid tag: {tag}")

        if safe_eval:
            results = verify_correctness(code=new_code, test=new_test)
        else:
            results = verify_correctness_unsafe(code=new_code, test=new_test)

        if results["status"] == "passed":
            self.code, self.function_name, self.function_call, self.test = new_code, new_function_name, new_function_call, new_test
        else:
            raise ValueError(f"Perturbation {tag} failed to preserve correctness: {results['traceback']}:\nOriginal Code:\n{self.code}\nOriginal Test:\n{self.test}\nPerturbed Code:\n{new_code}\nPerturbed Test:\n{new_test}")


    def get_prompt(self, task: Task, infer_mode: Mode) -> str:
        return get_run_prompt(
            dataset=self.dataset,
            task=task,
            infer_mode=infer_mode,
            code=self.code,
            function_name=self.function_name,
            function_call=self.function_call,
            output=self.output,
        ).strip()
    
    def get_incorrect_gen_prompt(self, task: Task) -> str:
        code = self.code
        expr = f"{self.function_call} == {repr(self.output)}"
        return get_incorrect_gen_prompt(code=code, expr=expr, task=task).strip()

        if task == Task.INPUT_PREDICTION:
            expression = self.function_call + " == " + repr(self.output)
        elif task == Task.OUTPUT_PREDICTION:
            expression = self.function_call
        else:
            raise ValueError("task must be either 'Task.INPUT_PREDICTION' or 'Task.OUTPUT_PREDICTION'")
        return get_incorrect_gen_prompt(code=self.code, expression=expression, task=task).strip()
