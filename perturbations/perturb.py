import ast
import threading
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from question.base import Question
from llm import LLMChat, Message
from generate.solutions import Solutions
from loader.base import QuestionDataset
from utils.format import Task, Perturbation
from evaluate import verify_correctness, verify_correctness_unsafe
from prompt import SYSTEM_PROMPT

io_lock = threading.Lock()

def extract_content(content: str, key: str) -> str:
    start_token = f"[{key}]"
    end_token = f"[/{key}]"
    return content.split(start_token)[-1].split(end_token)[0].strip()

def extract_answer(task: Task, content: str) -> str:
    if task == Task.INPUT_PREDICTION:
        if "incorrect_input" in content:
            return content.split("incorrect_input")[-1].split('=')[-1].strip()
    elif task == Task.OUTPUT_PREDICTION:
        if "incorrect_output" in content:
            return content.split("incorrect_output")[-1].split('=')[-1].strip()
    else:
        return None


def generate_incorrect_answer(
    model: LLMChat,
    question: Question,
    task: Task,
    safe_eval: bool = False,
    max_retry: int = 3,
) -> str:
    for _ in range(max_retry):
        prompt = question.get_incorrect_gen_prompt(task=task)
        inputs = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=prompt)
        ]
        responses = model.chat(messages=inputs, n=1)
        content = extract_content(responses[0], "EXPR")
        answer = extract_answer(task=task, content=content)
        temp_question = deepcopy(question)
        if task == Task.INPUT_PREDICTION:
            temp_question.function_call = f"{temp_question.function_name}({answer})"
            return_value = temp_question.function_call
        elif task == Task.OUTPUT_PREDICTION:
            temp_question.output = ast.literal_eval(answer)
            return_value = temp_question.output

        temp_question.test = f"assert {temp_question.function_call} == {repr(temp_question.output)}"
        
        if safe_eval:
            results = verify_correctness(code=temp_question.code, test=temp_question.test)
        else:
            results = verify_correctness_unsafe(code=temp_question.code, test=temp_question.test)

        if results["status"] == "failed":
            return return_value
        print(results)
        print(f"Retrying to generate incorrect answer for question {question.task_id}...")
    raise ValueError(f"Failed to generate incorrect answer for question {question.task_id} after {max_retry} attempts.")


def perturb_one_question(
    question: Question,
    perturbation: Perturbation,
    safe_eval: bool = False,
    **kwargs
) -> Solutions:
    if perturbation == Perturbation.MHC:
        if "task" not in kwargs or "model" not in kwargs:
            raise ValueError(f"Both 'task' and 'model' must be provided for {perturbation.value} perturbation.")
        task = kwargs.get("task")
        model = kwargs.get("model")
        incorrect_value = generate_incorrect_answer(model=model, question=question, task=task, safe_eval=safe_eval)
        kwargs["incorrect_value"] = incorrect_value    
    with io_lock:
        question.perturbate(tag=perturbation, safe_eval=safe_eval, **kwargs)


def perturb_dataset(
    dataset: QuestionDataset,
    perturbation: Perturbation,
    output_name: str,
    max_workers: int = 1,
    safe_eval: bool = False,
    **kwargs
) -> None:
    if not safe_eval:
        max_workers = 1  # Disable parallelism for unsafe evaluation
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                perturb_one_question,
                question,
                perturbation,
                safe_eval,
                **kwargs
            ): question
            for question in dataset.questions_list
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {perturbation.value} perturbation."):
            question = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred while processing question {question.task_id}: {e}")
    
    dataset.save(name=output_name)
    print(f"Perturbed dataset saved as {output_name}.jsonl")