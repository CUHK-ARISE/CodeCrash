import os
import json
from datetime import datetime

def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

def save_one_jsonl(data: dict, file_name: str) -> None:
    os.makedirs("./results", exist_ok=True)
    try:
        with open(f"./results/{file_name}.jsonl", "a") as file:
            file.write(json.dumps(data) + '\n')
    except:
        raise FileNotFoundError


def save_jsonl(data: list[dict], file_name: str, format: str = "w") -> None:
    os.makedirs("./results", exist_ok=True)
    try:
        with open(f"./results/{file_name}.jsonl", format) as file:
            for d in data:
                file.write(json.dumps(d) + '\n')
    except:
        raise FileNotFoundError


def append_one_to_json(data: dict, file_name: str) -> None:
    os.makedirs("./results", exist_ok=True)
    file_path = f"./results/{file_name}.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    existing_data.append(data)

    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4, default=json_serial)
        
        
def append_one_to_json_dict(data: dict, file_name: str) -> None:
    os.makedirs("./results", exist_ok=True)
    file_path = f"./results/{file_name}.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4, default=json_serial)


def save_json(data: list[dict], file_name: str) -> None:
    os.makedirs("./results", exist_ok=True)
    try:
        with open(f"./results/{file_name}.json", 'w') as file:
            json.dump(data, file, indent=4, default=json_serial)
    except:
        raise FileNotFoundError
