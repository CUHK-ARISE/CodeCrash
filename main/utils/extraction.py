import re
from typing import List, Tuple


def extract_code_snippet(text: str) -> str:
    """
    Extracts the code snippet enclosed within triple backticks (```...```) from a given text.
    """
    return text.split("```python")[-1].split("```")[0].strip()
# def extract_code_snippet(text: str) -> str:
#     """
#     Extracts the code snippet enclosed within triple backticks (```...```) from a given text.
#     """
#     text = text.replace("\\`\\`\\`", "```")
    
#     start_idx = text.find("```python")
#     if start_idx == -1:
#         return None

#     end_of_language_idx = text.find("\n", start_idx + 3)
#     if end_of_language_idx == -1:
#         return None
    
#     end_idx = text.find("```", end_of_language_idx)
#     if end_idx == -1:
#         return None

#     code_snippet = text[end_of_language_idx + 1:end_idx].strip()
    
#     return code_snippet


def extract_pattern(content: str, regex: str) -> Tuple[str, List[str]]:
    """
    Extract content that matches the given regex pattern from the input content,
    and return the content without the matched parts and the list of matched content.
    """
    # Find all matches of the pattern and remove the matched content from the content
    matched_content = re.findall(regex, content, re.DOTALL | re.MULTILINE)
    remove_content = re.sub(regex, "", content, flags=re.DOTALL | re.MULTILINE).strip()
    return remove_content.strip(), [c.strip() for c in matched_content]


def extract_json_format(content: str, key: str) -> str:
    """
    Extract the complete JSON object from the content.
    """
    # Updated regex to match key-value pairs including nested quotes
    # pattern = re.compile(r'(\{"' + re.escape(key) + r'":\s*"(.+?)"\s*\})', re.DOTALL)
    pattern = re.compile(r'(\{\s*"' + re.escape(key) + r'"\s*:\s*"(.*?)"\s*\})', re.DOTALL)
    match = pattern.search(content)
    return match.group(0) if match else None


def extract_content(content: str, key: str = "ANSWER") -> str:
    """
    Extract the content between the special tokens: [{key}] and [/{key}]
    """
    import ast
    
    if f"[{key}]" not in content and f"[/{key}]" not in content:
        return "No Answer"
    expression = content.split(f"[{key}]")[-1].split(f"[/{key}]")[0].strip()
    try:
        ast.parse(expression)
        return expression
    except:
        return "Syntax Error"
    
    # # Use f-string to build the regex pattern with the provided key
    # pattern = rf"\[{key}\](.*?)\[/{key}\]"
    # _, extracted_content = extract_pattern(content, pattern)

    # # Return the first match if available, otherwise None
    # return extracted_content[0].strip() if extracted_content else None


def extract_entry_code(content: str) -> str:
    """
    Extracts the function entry point from the given code content.
    """
    # Refine the regex to capture only the function signature
    code_snippet, entry_point_match = extract_pattern(content, r"def\s+[\w_]+\s*\([^)]*\)\s*:")
    return [match.strip() for match in entry_point_match] if entry_point_match else None


def extract_entry_point(assert_statement: str) -> str:
    """
    Extract the function name (xxx) from an assert statement.
    """
    # Remove any built-in wrappers like set(), list(), dict(), math.isclose()
    cleaned_statement = re.sub(r'\b(set|list|dict)\((.*?)\)', r'\2', assert_statement)

    # Extract the first custom function name in the cleaned statement, accounting for 'not' if present
    match = re.search(r'assert\s+(?:not\s+)?([\w_]+)\s*\(', cleaned_statement)
    return match.group(1) if match else None


def extract_function_name_from_definition(function_def: str) -> str:
    """
    Extract the function name from a function definition.
    """
    # Regex to match the function name after 'def'
    match = re.search(r'\bdef\s+([\w_]+)\s*\(', function_def)
    return match.group(1) if match else None


def search_match_function(functions: List[str], entry_point: str) -> str:
    """
    Search for the function matching the entry point exactly
    """
    matching_function = [
        match for match in functions if re.search(rf'\bdef\s+{entry_point}\s*\(', match)
    ]
    return matching_function[0] if matching_function else None
