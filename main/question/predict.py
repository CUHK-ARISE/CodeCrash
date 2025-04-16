from typing import List
from llm import LLMChat, Message
from typing import List, Optional

from question import BaseQuestion, Solution, parse_example
from question.modification.cctest import process_RPV, process_REF, process_RTF, process_GRA, process_GGV
from question.modification.comments import process_MDC, process_MPS
from evaluators import verify_correctness
from utils import extract_content, console_output
from prompt.prediction_prompts import RUN_PREDICTION_INPUT, RUN_PREDICTION_OUTPUT, RUN_PREDICTION_INPUT_COT, RUN_PREDICTION_OUTPUT_COT
from prompt.prompts import GENERATE_INCORRECT_INPUT, GENERATE_INCORRECT_OUTPUT


class PredictionQuestion(BaseQuestion):
    SYSTEM_PROMPT = "You are a helpful assistant. Do NOT output any extra information."

    def __init__(self, task_id: str, dataset: str, function_name: str, code: str,
                 function_call: str, output: str, expression: str = None, output_format: dict = {}) -> None:
        super().__init__(task_id, dataset)
        self.function_name = function_name
        self.code = code
        self.function_call = function_call
        self.output = output
        self.expression_format = parse_example(function_name, f"{self.function_call} == {self.output}")
        self.expression = self.expression_format.get_expression() if expression is None else expression
        self.output_format = output_format

    def to_dict(self) -> dict:
        state = self.__dict__.copy()
        state["expression_format"] = self.expression_format.to_dict()
        return state
    
    @classmethod
    def from_dict(cls, state: dict) -> "PredictionQuestion":
        state.pop("expression_format")
        return cls(
            **state,
        )
        
    def modify_expression(self, new_expression: str, new_function_name: str = None) -> None:
        function_name = new_function_name if new_function_name else self.function_name
        self.expression_format = parse_example(function_name, new_expression)
        self.expression = self.expression_format.get_expression()
        self.function_call = self.expression_format.get_function_call()
        self.function_name = self.expression_format.func_name
    

    def get_run_prompt(self, input_dict: dict[str:str], mode: str, cot: bool = False) -> str:
        if mode == "input":
            if cot:
                return RUN_PREDICTION_INPUT_COT.format(**input_dict)
            else:
                return RUN_PREDICTION_INPUT.format(**input_dict)
        elif mode == "output":
            if cot:
                return RUN_PREDICTION_OUTPUT_COT.format(**input_dict)
            else:
                return RUN_PREDICTION_OUTPUT.format(**input_dict)
    
    
    def run(self, model: LLMChat, mode: str, **kwargs) -> List[Solution]:
        def extract_answer(expression: str) -> str:
            try:
                example = parse_example(self.function_name, expression)
                if mode == "input":
                    return example.get_function_call()
                elif mode == "output":
                    return example.output
            except Exception as e:
                return "Syntax Error"
        
        input_dict = {
            "programming_language": "python",
            "code": self.code,
            "input": self.function_call,
            "output": self.output,
            "function_name": self.function_name,
        }

        cot = kwargs.get("cot", False)
        input_prompt = self.get_run_prompt(input_dict, mode, cot)
        inputs = [Message(role="system", content=self.SYSTEM_PROMPT),
                  Message(role="user", content=input_prompt)]

        solutions = []
        responses, usage = self.request(model, inputs, **kwargs)
        for response in responses:
            extracted_content = extract_content(response)
            if extracted_content is not None:
                answer = extract_answer(extracted_content)
                if answer is None:
                    answer = "Syntax Error"
            else:
                answer = "No Answer"
            solutions.append((response, answer))
        
        return Solution(self.task_id, self.dataset, input_prompt, solutions, self.output_format, usage=usage)
    
    
    """
    CCTest methods
    """
    def verify_modified_code(self, new_code: str, expression: str, flag: str) -> bool:
        passed, e = verify_correctness(expression, new_code)
        if passed:
            self.code = new_code
            return True
        else:
            raise ValueError(f"{e}\n {self.task_id}: The code is incorrect after processing {flag}.\n{self.code}\n=======\n{new_code}\n{expression}")
    
    def process_RPV(self) -> None:
        new_code, new_expression = process_RPV(self.code, self.function_name, self.expression)
        self.verify_modified_code(new_code, new_expression, "RPV")
        self.modify_expression(new_expression, self.function_name)

    def process_REF(self) -> None:
        new_code, new_function_name, new_function_call = process_REF(self.code, self.function_name, self.function_call)
        new_expression = f"{new_function_call} == {self.output}"
        self.verify_modified_code(new_code, new_expression, "REF")
        self.modify_expression(new_expression, new_function_name)
    
    def process_RTF(self) -> None:
        new_code = process_RTF(self.code)
        self.verify_modified_code(new_code, self.expression, "RTF")

    def process_GRA(self) -> None:
        new_code = process_GRA(self.code)
        self.verify_modified_code(new_code, self.expression, "GRA")
    
    def process_GGV(self) -> None:
        new_code = process_GGV(self.code)
        self.verify_modified_code(new_code, self.expression, "GGV")
    
    def process_structural_perturbation(self, flag: Optional[List | str]) -> None:
        if isinstance(flag, str):
            flag = [flag]
        
        for f in flag:
            if f in ["RPV", "REF", "RTF", "IRR", "GRA", "GGV"]:
                method = getattr(self, f"process_{f}", None)
                if method and callable(method):
                    method()
            elif f == "REN":    # Renaming Entities (Functions, Parameters, and Variables)
                self.process_REF()
                self.process_RPV()
                return
            elif f == "GBC":    # Garbage Code
                self.process_GGV()
                self.process_GRA()
                return
            elif f == "ALL":
                self.process_GGV()
                self.process_RTF()
                self.process_GRA()
                self.process_REF()
                self.process_RPV()
                return
            else:
                raise ValueError(f"Invalid flag: {flag}")
    
    
    """
    Comments methods
    """
    def generate_incorrect_expression(self, model: LLMChat, mode: str) -> str:
        
        input_dict = {
            "programming_language": "python",
            "code": self.code,
            "function_name": self.function_name,
            "expression": self.expression
        }
        
        if mode == "input":
            input_prompt = GENERATE_INCORRECT_INPUT.format(**input_dict)
        elif mode == "output":
            input_prompt = GENERATE_INCORRECT_OUTPUT.format(**input_dict)

        inputs = [Message(role="system", content=self.SYSTEM_PROMPT),
                  Message(role="user", content=input_prompt)]

        while True:
            response = self.request(model, inputs, n=1)[0][0]
            new_expression = extract_content(response, "EXPRESSION")
            passed, e = verify_correctness(new_expression, self.code)
            
            if not passed:
                new_example = parse_example(self.function_name, new_expression, "incorrect")
                new_function_call = new_example.get_function_call()
                new_output = new_example.output
                print(new_output)
                
                if mode == "input" and new_output == self.output:
                    return new_example
                elif mode == "output" and new_function_call == self.expression_format.get_function_call():
                    return new_example
                else:
                    msg = "input arguments" if mode == "output" else "output value"
                    inputs += [Message(role="assistant", content=response)]
                    inputs += [Message(role="user", content=f"Do not modify the {msg}. Please do again.")]
            else:
                inputs += [Message(role="assistant", content=response)]
                inputs += [Message(role="user", content="The expression is correct, but we want you to modify the input and make it incorrect. Please do again.")]
            console_output(f"{self.task_id}: Failed to generate incorrect expression.\n{self.code}\n{new_expression}")
    
    
    def process_textual_perturbation(self, flag: str, **kwargs) -> str:
        once = kwargs.get("once", False)
        p = kwargs.get("p", 1)
        
        if flag == "MDC":
            new_code = process_MDC(self.code, self.output, once, p)
            passed, e = verify_correctness(self.expression, new_code)
            if passed:
                self.code = new_code
                return new_code
            else:
                raise ValueError(f"{e}\n {self.task_id}: The code is incorrect after adding comment.\n{new_code}")
        
        elif flag == "MPS":
            new_code = process_MPS(self.code, self.output)
            passed, e = verify_correctness(self.expression, new_code)
            if passed:
                self.code = new_code
                return new_code
            else:
                raise ValueError(f"{e}\n {self.task_id}: The code is incorrect after adding comment.\n{new_code}")
            