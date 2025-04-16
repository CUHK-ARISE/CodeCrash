import os
import traceback
from typing import Optional
from copy import deepcopy
from tqdm.auto import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from loader import Crux, LiveCodeBench


def multiprocess_run(
    data_list,
    process_function,
    max_workers=5,
    progress_bar_desc="Processing",
    additional_args=None
):
    additional_args = additional_args or {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_function, item, **additional_args)
            for item in data_list
        ]

        for future in tqdm(as_completed(futures), total=len(data_list), desc=progress_bar_desc):
            try:
                future.result()
            except Exception as e:
                print(f"Error during processing: {e}")
                traceback.print_exc()

class Runner:
    def __init__(self, dataset, model) -> None:
        """
        Initialize the server with a dataset and model.
        """
        self.dataset = dataset
        self.model = model
        self.folder = model.folder_name
        self.lock = Lock()
        if type(self.dataset) in [Crux, LiveCodeBench]:
            self.format = "json"

    def run(self, file_name: str = "save", max_workers: int = 30, **kwargs) -> None:
        """
        Process the dataset with multi-threading.
        """
        n = kwargs.pop("n", 1)
        os.makedirs(f"results/{self.folder}", exist_ok=True)
        results_path = f"results/{self.folder}/{file_name}.{self.format}"
        rest_count = self.dataset.load_saved_execution(results_path, n)
        
        rest_questions = [
            (question, rest_count[question.task_id])
            for question in self.dataset.questions_list
            if rest_count[question.task_id] > 0
        ]
        print(rest_questions)

        def process_function(question_tuple, **kwargs):
            question, n = question_tuple
            try:
                solutions = question.run(self.model, n=n, **kwargs)
                with self.lock:
                    solutions.save(f"results/{self.folder}/{file_name}")
                return question
            except Exception as e:
                print(
                    f"Error in process_function for question {question.task_id}: {e}")
                traceback.print_exc()
                return None

        multiprocess_run(
            data_list=rest_questions,
            process_function=process_function,
            max_workers=max_workers,
            progress_bar_desc=f"{self.model.model_name}: Processing Questions",
            additional_args=kwargs
        )

    def generate_examples(self, file_name: str, n_examples: int = 1, **kwargs):
        """
        Generate examples for each question in the dataset using multiprocessing.
        """
        copy_dataset = deepcopy(self.dataset)
        for question in tqdm(copy_dataset.questions_list):
            for _ in range(n_examples):
                question.generate_example(self.model, **kwargs)
            copy_dataset.save(file_name)
        return copy_dataset
    
    def process_structural_perturbation(self, file_name: Optional[str | None], flag: str) -> None:
        """
        Applies a specified method to all questions in the dataset and saves the result.
        """
        copy_dataset = deepcopy(self.dataset)
        for question in copy_dataset.questions_list:
            question.process_structural_perturbation(flag)
        if file_name is not None:
            copy_dataset.save(file_name)
        return copy_dataset
    
    def process_textual_perturbation(self, file_name: Optional[str | None], flag: str, **kwargs) -> None:
        """
        Adds comments to the questions in the dataset.
        """
        copy_dataset = deepcopy(self.dataset)
        for question in copy_dataset.questions_list:
            question.process_textual_perturbation(flag, **kwargs)
        if file_name is not None:
            copy_dataset.save(file_name)
        return copy_dataset
    