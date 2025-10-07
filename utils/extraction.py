import ast
from typing import Optional

def extract_code_snippet(text: str) -> str:
    """
    Extracts the code snippet enclosed within triple backticks (```...```) from a given text.
    """
    if "```python" in text:
        return text.split("```python")[-1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[-1].split("```")[0].strip()
    else:
        raise ValueError("No code snippet (```) found in the text.")


def extract_content(text: str, key: str) -> Optional[str]:
    """
    Extract the content between the special tokens: [{key}] and [/{key}]
    """
    if f"[{key}]" not in text and f"[/{key}]" not in text:
        return None
    try:
        content = text.split(f"[{key}]")[-1].split(f"[/{key}]")[0].strip()
        ast.parse(content)
        return content
    except SyntaxError as e:
        return None
