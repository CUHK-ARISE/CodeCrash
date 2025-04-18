import gzip
import json
from typing import List


def load_file(file_path: str, key: str) -> List:
    try:
        if file_path.endswith("json"):
            with open(file_path, "r") as file:
                data = json.load(file)
        elif file_path.endswith("jsonl"):
            with open(file_path, "r") as file:
                data = [json.loads(line) for line in file]
        elif file_path.endswith("jsonl.gz"):
            with gzip.open(file_path, 'rt') as file:
                data = [json.loads(line) for line in file]
        else:
            raise ValueError(f"Unsupported file extension for file: {file_path}")
        return {item[key]: item for item in data}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: The file at {file_path} was not found.")


# def jsonl_to_dict(data: list, key: str) -> dict:
#     """
#     Convert a JSONL list to a dictionary based on a specified key.
#     """
#     try:
#         return {item[key]: item for item in data}
#     except KeyError:
#         raise KeyError(
#             f"Key '{key}' does not exist in one of the JSON objects.")
