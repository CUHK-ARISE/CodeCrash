import threading
from tqdm import tqdm
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt import SYSTEM_PROMPT
from loader import QuestionDataset
from llm import LLMChat, Message
from question.base import Question
from generate.solutions import Solutions
from utils.format import Task, Mode
from utils.save import save_sol, load_existing_solutions
from utils.extraction import extract_content


def extract_answer(task: Task, content: str) -> Optional[str]:
    content = extract_content(content, key="ANSWER")
    if content is None:
        return None
    try:
        contents = content.split("==")
        if task == Task.INPUT_PREDICTION:
            return contents[0].strip()
        elif task == Task.OUTPUT_PREDICTION:
            return contents[1].strip()
    except Exception as e:
        return None


def process_one_question(
    model: LLMChat,
    question: Question,
    task: Task,
    infer_mode: Mode,
    n: int = 1,
) -> Solutions:
    prompt = question.get_prompt(task=task, infer_mode=infer_mode)
    inputs = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=prompt)
    ]
    responses = []
    answers = []
    while len(answers) < n:
        responses += model.chat(messages=inputs, n=n-len(answers))
        answers += [extract_answer(task=task, content=response) for response in responses]
    
    solutions = Solutions(question=question, input_prompt=prompt, responses=responses, solutions=answers)
    return solutions


def process_questions(
    model: LLMChat,
    questions: List[Tuple[Question, int, Solutions]],
    task: Task,
    infer_mode: Mode,
    filepath: str,
) -> None:
    for question, n, exist_solutions in tqdm(questions, desc=f"Processing {model.model_name} -> {filepath}"):
        try:
            solutions = process_one_question(
                model=model,
                question=question,
                task=task,
                infer_mode=infer_mode,
                n=n,
            )
            new_solutions = Solutions.merge(sol1=exist_solutions, sol2=solutions)
            save_sol(solutions=new_solutions, filepath=filepath)
        except Exception as e:
            print(f"[Error on question {question.task_id}]: {e}")


def process_questions_parallel(
    model: LLMChat,
    questions: List[Tuple[Question, int, Solutions]],
    task: Task,
    infer_mode: Mode,
    filepath: str,
    max_workers: int = 8,
) -> None:
    
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_question,
                model, question, task, infer_mode, n
            ): (question, n, exist_solutions)
            for question, n, exist_solutions in questions
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model.model_name} -> {filepath}"):
            question, n, exist_solutions = futures[future]
            # try:
            solutions = future.result()
            new_solutions = Solutions.merge(sol1=exist_solutions, sol2=solutions)
            with lock:
                save_sol(solutions=new_solutions, filepath=filepath)
            # except Exception as e:
            #     print(f"[Error on question {question.task_id}]: {e}")


def process_dataset(
    model: LLMChat,
    dataset: QuestionDataset,
    task: Task,
    infer_mode: Mode,
    filepath: str,
    n: int = 1,
    load_existing: bool = True,
    max_workers: int = 5,
) -> None:
    questions = []

    if load_existing:
        existing_solutions = load_existing_solutions(filepath=filepath)
    else:
        existing_solutions = {}
    
    questions = []
    for question in dataset.questions_list:
        if question.task_id in existing_solutions:
            exist_solutions = Solutions(
                question=question,
                input_prompt=existing_solutions[question.task_id].get("input_prompt", ""),
                responses=existing_solutions[question.task_id].get("responses", []),
                solutions=existing_solutions[question.task_id].get("solutions", []),
            )
            to_generate = max(0, n - len(exist_solutions.solutions))
            if to_generate > 0:
                questions.append((question, to_generate, exist_solutions))
        else:
            exist_solutions = Solutions(question=question)
            questions.append((question, n, exist_solutions))

    process_questions_parallel(model=model, questions=questions, task=task, infer_mode=infer_mode, filepath=filepath, max_workers=max_workers)