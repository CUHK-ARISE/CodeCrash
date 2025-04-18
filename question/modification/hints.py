import ast
import string
import random
from typing import Any, List

"""
Randomization Utilities for Strings and Primitives
"""
def randomize_str(value: str, n: int) -> str:
    """
    Randomizes a string by modifying up to `n` characters or generating a random string if empty.
    """
    if not value:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

    value = list(value)
    indices = random.sample(range(len(value)), min(n, len(value)))

    for idx in indices:
        char = value[idx]
        if char.isdigit():
            value[idx] = random.choice(string.digits)
        elif char.islower():
            value[idx] = random.choice(string.ascii_lowercase)
        elif char.isupper():
            value[idx] = random.choice(string.ascii_uppercase)
        else:
            repeat_count = random.randint(2, 3)
            value[idx] = char * repeat_count

    return ''.join(value)


def randomize_primitive(value: Any, randomness: float = 1) -> Any:
    """
    Randomizes a primitive value by modifying its content or flipping its value.
    """
    if isinstance(value, bool):
        return not value

    elif isinstance(value, int):
        sign = -1 if value < 0 else 1
        num_digits = max(1, len(str(abs(value))))
        rand_val = random.randint(10**(num_digits - 1), 10**num_digits - 1)
        return sign * rand_val

    elif isinstance(value, float):
        sign = -1 if value < 0 else 1
        int_part_digits = max(1, len(str(int(abs(value)))))
        rand_val = random.uniform(10**(int_part_digits - 1), 10**int_part_digits - 1)
        return sign * round(rand_val, 2)

    elif isinstance(value, str):
        n = max(1, int(len(value) * randomness)) if len(value) > 1 else max(1, int(10 * randomness))
        return randomize_str(value, n)

    elif isinstance(value, bytes):
        return bytes(random.choices(range(256), k=len(value)))

    elif value is None:
        return random.choice([False, 0, "", [], {}, None])

    else:
        print(f"Unsupported type: {type(value)}")
        return value


def get_useless_value(value_type: Any, random=True) -> Any:
    """
    Returns a useless value for the given type along with a description.
        - value_type (Any): The type for which to generate a useless value.
    """
    if value_type == bool:
        return False
    elif value_type == int:
        return 0
    elif value_type == float:
        return 0.0
    elif value_type == str:
        return ""
    elif value_type == bytes:
        return b""
    elif value_type == list:
        return [None]
    elif value_type == tuple:
        return (None)
    elif value_type == set:
        return set(None)
    elif value_type == dict:
        return {None}
    return None
    

"======================================================================================================"
"""
Primary Functions for Modifying Code
"""
def add_output_comments(code: str, target_function: str, value: Any) -> str:
    """
    Add comments to the return statement of a specific function in the code.
    """
    value_repr = repr(value)
    tree = ast.parse(code)
    lines = code.splitlines(True)
    
    target_function_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == target_function),
        None
    )
    
    if not target_function_node:
        raise ValueError(f"Function '{target_function}' not found in the provided code.")
    
    return_nodes = [node for node in ast.walk(target_function_node) if isinstance(node, ast.Return)]
    
    if len(return_nodes) == 1:
        # Single return node
        return_node = return_nodes[0]
        return_line_index = return_node.lineno - 1
        lines[return_line_index] = lines[return_line_index].rstrip('\n')
        lines[return_line_index] += f"    # The return value is {value_repr}\n"
    
    else:
        # Multiple return nodes or no return nodes
        all_function_nodes = list(ast.walk(target_function_node))
        last_lineno = max(getattr(node, 'lineno', 0) for node in all_function_nodes)
        lines.insert(last_lineno, f"\n    # The return value is {value_repr}\n")

    return ''.join(lines)


def add_input_hint(code: str, target_function: str, message: str) -> str:
    """
    Add an incorrect hint to the input of a specific function in the code.
    """
    class FunctionHintInserter(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == target_function:
                node.decorator_list.insert(0, ast.Expr(value=ast.Str(s=f"{message}")))
            return node
    
    code = ast.unparse(ast.parse(code))
    tree = ast.parse(code)
    transformer = FunctionHintInserter()
    new_tree = transformer.visit(tree)
    
    # Convert the modified AST back to source code
    lines = code.split('\n')
    for node in ast.walk(new_tree):
        if isinstance(node, ast.FunctionDef):
            line_index = node.lineno - 1
            lines[line_index] = lines[line_index] + f"    # {message}"
    
    return "\n".join(lines)


def randomize_output_structure(node: ast.AST, randomness: float, nested: bool = False):
    """
    Recursively modifies an AST node by randomly perturbing constants and structures
    like lists, tuples, sets, and dictionaries based on a randomness probability.

    Args:
        node (ast.AST): The AST node to modify.
        randomness (float): Probability for each mutation to occur (0.0 to 1.0).
        nested (bool): Whether the current node is nested within another structure.
    """
    def should_modify() -> bool:
        return random.random() <= randomness
    
    def partial_shuffle(lst, t):
        if len(lst) < 2 or t <= 0:
            return lst
        n_swaps = max(1, int(len(lst) * t))
        for _ in range(min(n_swaps, len(lst) * (len(lst) - 1) // 2)):
            i, j = random.sample(range(len(lst)), 2)
            lst[i], lst[j] = lst[j], lst[i]
        return lst
    
    # Handle list-like structures
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if node.elts:
            if not nested and should_modify():
                if len(node.elts) == 1 or random.random() < 0.5:
                    # Add a new element by mutating a sampled one
                    elem = random.choice(node.elts)
                    node.elts.append(elem)
                else:
                    # Remove an element
                    node.elts.pop(random.randint(0, len(node.elts) - 1))
            
            if should_modify():
                node.elts = partial_shuffle(node.elts, randomness)

        # Recurse into elements
        for i, elem in enumerate(node.elts):
            if isinstance(elem, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                randomize_output_structure(elem, randomness, nested=True)
            elif isinstance(elem, ast.Constant) and should_modify():
                node.elts[i] = ast.Constant(value=randomize_primitive(elem.value, randomness))

        # Ensure non-empty
        if not node.elts:
            node.elts.append(ast.Constant(value=randomize_primitive(None)))

    # Handle dictionaries
    elif isinstance(node, ast.Dict):
        if node.keys:
            # Structural modification for dict (add or delete one key-value pair)
            if not nested and should_modify():
                
                if len(node.keys) == 1 or random.random() < 0.5:
                    # Add a new key-value pair by mutating a sampled one
                    key_sample = node.keys[0]
                    value_sample = node.values[0]
                    if isinstance(key_sample, ast.Constant) and isinstance(value_sample, ast.Constant):
                        new_key = randomize_primitive(key_sample.value, randomness)
                        if isinstance(value_sample, ast.Constant):
                            new_value = randomize_primitive(value_sample.value, randomness)
                        else:
                            new_value = randomize_primitive(None, randomness)
                        node.keys.append(ast.Constant(value=new_key))
                        node.values.append(ast.Constant(value=new_value))
                else:
                    # Remove a key-value pair
                    idx = random.randint(0, len(node.keys) - 1)
                    del node.keys[idx]
                    del node.values[idx]

        # Recurse into values
        for i, elem in enumerate(node.values):
            if isinstance(elem, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                randomize_output_structure(elem, randomness, nested=True)
            elif isinstance(elem, ast.Constant) and should_modify():
                node.values[i] = ast.Constant(value=randomize_primitive(elem.value, randomness))

        # Ensure non-empty
        if not node.keys:
            node.keys.append(ast.Constant(value=randomize_primitive("")))
            node.values.append(ast.Constant(value=randomize_primitive(None)))

    # Handle unary minus, like: -3 → randomize the 3 part
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and not nested:
            perturbed = randomize_primitive(node.operand.value, randomness)
            node.operand = ast.Constant(value=abs(perturbed))

    # Handle simple constants directly
    elif isinstance(node, ast.Constant):
        node.value = randomize_primitive(node.value, randomness)


def random_modify_output(original_output: str, follow_structure: bool = True, randomness: float = 0.5) -> Any:
    """Modify one element in the original output structure."""
    if follow_structure:
        while True:
            tree = ast.parse(original_output, mode='eval')
            randomize_output_structure(tree.body, randomness)
            modified_value = ast.unparse(tree.body)
            if modified_value != original_output:
                return modified_value
    else:
        modified_value = repr(get_useless_value(type(ast.literal_eval(original_output), True)))
        if modified_value != original_output:
            return modified_value
    
    