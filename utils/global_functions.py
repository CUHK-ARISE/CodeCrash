import re
from typing import List
from colorama import Fore, Style


def count_lines(content: str) -> int:
    """
    Counts the number of lines in content string.
    """
    lines = content.splitlines()
    return len(lines)


def check_lines(content: str, lines: int) -> bool:
    """
    Check if the input content has exactly the specified number of lines.
    """
    if count_lines(content) == lines:
        return True
    else:
        return False


# Get Prompt Template
def get_prompt_template(file_path: str, inputs: dict[str:str]) -> str:
    with open(file_path, 'r') as file:
        prompt_template = file.read().split("<commentblockmarker>###</commentblockmarker>")[1].strip()
    return prompt_template.format(**inputs)


# def get_prompt_template(file_path: str, inputs: List) -> str:
#     with open(file_path, 'r') as file:
#         generated_prompt = file.read().split("<commentblockmarker>###</commentblockmarker>")[1].strip()
#     for index, item in enumerate(inputs):
#         key = f"!<INPUT {index+1}>!"
#         generated_prompt = generated_prompt.replace(key, str(item))
#     return generated_prompt


def replace_entry_point(text, entry_point, new_entry_point):
    # Match only exact occurrences of entry_point, not part of longer words
    text = re.sub(rf'\b{entry_point}\b\s*\(', f'{new_entry_point}(', text)
    text = re.sub(rf'`{entry_point}`', f'`{new_entry_point}`', text)
    return text
