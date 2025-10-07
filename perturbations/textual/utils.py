import ast
import random
import string
from typing import Any, Tuple

def randomize_str(value: str, n: int) -> str:
    """
    Randomizes a string by modifying up to `n` characters or generating a random string if no input is provided.
    """
    if not value:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=n))
    n = min(n, len(value))
    value_list = list(value)
    indices = random.sample(range(len(value_list)), n)
    for idx in indices:
        ch = value_list[idx]
        if ch.isdigit():
            value_list[idx] = random.choice(string.digits)
        elif ch.isalpha() and ch.islower():
            value_list[idx] = random.choice(string.ascii_lowercase)
        elif ch.isalpha() and ch.isupper():
            value_list[idx] = random.choice(string.ascii_uppercase)
        else:
            value_list[idx] = ch * random.randint(1, 2)
    return ''.join(value_list)


def randomize_primitive(value: Any, n: int = 1) -> Any:
    """
    Randomizes a primitive value by changing its type or modifying its content.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        num_digits = max(1, len(str(abs(value))))
        lo = 10 ** (num_digits - 1)
        hi = 10 ** num_digits - 1
        return random.randint(lo, hi)
    if isinstance(value, float):
        num_digits = max(1, len(str(int(abs(value))) or "0"))
        lo = 10 ** (num_digits - 1)
        hi = 10 ** num_digits - 1
        return round(random.uniform(lo, hi), 2)
    if isinstance(value, str):
        return randomize_str(value, n)
    if isinstance(value, bytes):
        return bytes(random.choices(range(256), k=len(value)))
    if value is None:
        return random.choice([False, 0])
    return value


def get_useless_value(value_type: Any) -> Tuple[Any, str]:
    """
    Returns a useless value for the given type along with a brief description.
    """
    if value_type == bool:
        return False, "False"
    if value_type == int:
        return 0, "zero"
    if value_type == float:
        return 0.0, "zero"
    if value_type == str:
        return "", "an empty string"
    if value_type == bytes:
        return b"", "an empty bytes"
    if value_type == list:
        return [], "an empty list"
    if value_type == tuple:
        return (), "an empty tuple"
    if value_type == set:
        return set(), "an empty set"
    if value_type == dict:
        return {}, "an empty dictionary"
    return None, "None"


# ---------------------------
# Output structure mutation
# ---------------------------
def _randomize_output_structure(node: ast.AST, threshold: float, nested: bool = False):
    """Internal: recursively randomize containers / constants in an AST expression node."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        # duplicate an element
        if node.elts and not nested and random.random() < threshold:
            node.elts.append(random.choice(node.elts))
            random.shuffle(node.elts)
        # delete an element
        if len(node.elts) > 1 and not nested and random.random() < threshold:
            node.elts.pop(random.randrange(len(node.elts)))
            random.shuffle(node.elts)
        # recurse
        for i, elem in enumerate(node.elts):
            if isinstance(elem, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                _randomize_output_structure(elem, threshold, nested=True)
            elif isinstance(elem, ast.Constant) and random.random() < threshold:
                node.elts[i] = ast.Constant(value=randomize_primitive(elem.value))
        if len(node.elts) == 0:
            node.elts.append(ast.Constant(value=randomize_primitive(None)))
    
    elif isinstance(node, ast.Dict):
        # add kv
        if node.keys and not nested and random.random() < threshold:
            base_key = node.keys[0].value if isinstance(node.keys[0], ast.Constant) else ""
            base_val = node.values[0].value if isinstance(node.values[0], ast.Constant) else ""
            node.keys.append(ast.Constant(value=randomize_primitive(base_key)))
            node.values.append(ast.Constant(value=randomize_primitive(base_val)))
        # delete kv
        if len(node.keys) > 1 and not nested and random.random() < threshold:
            idx = random.randrange(len(node.keys))
            del node.keys[idx]
            del node.values[idx]
        # recurse
        for i, val in enumerate(node.values):
            if isinstance(val, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                _randomize_output_structure(val, threshold, nested=True)
            elif isinstance(val, ast.Constant) and random.random() < threshold:
                node.values[i] = ast.Constant(value=randomize_primitive(val.value))
        if len(node.keys) == 0:
            node.keys.append(ast.Constant(value=randomize_primitive("")))
            node.values.append(ast.Constant(value=randomize_primitive("")))
    
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and not nested:
            node.operand = ast.Constant(value=randomize_primitive(-node.operand.value))
    
    elif isinstance(node, ast.Constant) and not nested:
        node.value = randomize_primitive(node.value)


def random_modify_output(original_output: str, follow_structure: bool = True, threshold: float = 0.5) -> str:
    """
    Modify one element in the original output structure (string containing a Python literal).
    """
    if follow_structure:
        expr_src = original_output
    else:
        ty = type(ast.literal_eval(original_output))
        expr_src = repr(get_useless_value(ty)[0])
    
    tree = ast.parse(expr_src, mode="eval")
    _randomize_output_structure(tree.body, threshold)
    return ast.unparse(tree.body)