import ast
import string
import random

from typing import Any, List
from prompt.misleading_comments import *

"""
Randomization Utilities for Strings and Primitives
"""

def randomize_str(value: str, n: int) -> str:
    """
    Randomizes a string by modifying up to `n` characters or generating a random string if no input is provided.
        - value (str): The input string to modify. If None, generates a random string
        - n (int): The number of characters to modify in the input string.
    """
    if not value:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=n))
    else:
        n = min(n, len(value))
        value = list(value)
        indices = random.sample(range(len(value)), n)
        
        for index in indices:
            char = value[index]
            if char.isdigit():
                value[index] = random.choice(string.digits)
            elif char.isalpha() and char.islower():
                    value[index] = random.choice(string.ascii_lowercase)
            elif char.isalpha() and char.isupper():
                value[index] = random.choice(string.ascii_uppercase)
            else:
                repeat_count = random.randint(1, 2)
                value[index] = char * repeat_count
        
        return ''.join(value)


def randomize_primitive(value: Any, n: int = 1) -> Any:
    """
    Randomizes a primitive value by changing its type or modifying its content.
        - value (Any): The input value to randomize.
        - n (int, optional): The number of characters to modify in a string. Defaults to 1.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        num_digits = len(str(value))
        return random.randint(10**(num_digits - 1), 10**num_digits - 1)
    elif isinstance(value, float):
        num_digits = len(str(value).split('.')[0])
        random_value = random.uniform(10**(num_digits - 1), 10**num_digits - 1)
        return round(random_value, 2)
    elif isinstance(value, str):
        return randomize_str(value, n)
    elif isinstance(value, bytes):
        return bytes(random.choices(range(256), k=len(value)))
    elif value is None:
        return random.choice([False, 0,])


def get_useless_value(value_type: Any) -> Any:
    """
    Returns a useless value for the given type along with a description.
        - value_type (Any): The type for which to generate a useless value.
    """
    if value_type == bool:
        return False, "False"
    elif value_type == int:
        return 0, "zero"
    elif value_type == float:
        return 0.0, "zero"
    elif value_type == str:
        return "", "an empty string"
    elif value_type == bytes:
        return b"", "an empty bytes"
    elif value_type == list:
        return [], "an empty list"
    elif value_type == tuple:
        return (), "an empty tuple"
    elif value_type == set:
        return set(), "an empty set"
    elif value_type == dict:
        return {}, "an empty dictionary"
    return None, "None"
    

"======================================================================================================"

"""
Primary Functions for Modifying Code
"""
def search_returns(code: str) -> List[ast.Return]:
    """
    Searches for all return nodes in the given code.
    """
    tree = ast.parse(code)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Return)]


def add_output_comments(code: str, target_function: str, output_value: Any) -> str:
    """
    Adds comments indicating the return value to the target function in the code.
        - code (str): The code snippet to modify.
        - target_function (str): The name of the function to modify.
        - output_value (Any): The return value to add as a comment.
    """
    output_value_repr = repr(output_value)
    tree = ast.parse(code)
    lines = code.splitlines(True)
    
    target_function_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == target_function),
        None
    )
    
    if not target_function_node:
        raise ValueError(f"Function '{target_function}' not found in the provided code.")
    
    return_nodes = [node for node in ast.walk(target_function_node) if isinstance(node, ast.Return)]
    
    if len(return_nodes) == 1:  # Single return node
        return_node = return_nodes[0]
        return_line_index = return_node.lineno - 1
        lines[return_line_index] = lines[return_line_index].rstrip('\n')
        lines[return_line_index] += f"    # The return value is {output_value_repr}\n"
    
    else:  # Multiple return nodes or no return nodes
        all_function_nodes = list(ast.walk(target_function_node))
        last_lineno = max(getattr(node, 'lineno', 0) for node in all_function_nodes)
        lines.insert(last_lineno, f"\n    # The return value is {output_value_repr}\n")

    return ''.join(lines)


def add_input_hint(code: str, target_function: str, message: str) -> str:
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


def randomize_output_structure(node: ast.AST, threshold: float, nested: bool = False):
    """
    Modify one element in the AST structure by performing randomized operations on lists, sets, tuples, or dictionaries.
        - node (ast.AST): The root AST node to modify.
        - threshold (float): A probability threshold for modifying constant values.
        - nested (bool): A flag indicating whether the current node is nested inside another structure.
    """
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        # Randomly duplicate one element
        if node.elts and not nested and random.random() < threshold:
            element_to_duplicate = random.choice(node.elts)
            node.elts.append(element_to_duplicate)
            random.shuffle(node.elts)

        # Randomly delete an element
        if len(node.elts) > 1 and not nested and random.random() < threshold:
            node.elts.pop(random.randint(0, len(node.elts) - 1))
            random.shuffle(node.elts)

        # Recursive modification for elements that are lists, sets, tuples, or dictionaries
        for i, elem in enumerate(node.elts):
            if isinstance(elem, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                randomize_output_structure(elem, threshold, nested=True)
            elif isinstance(elem, ast.Constant) and random.random() < threshold:
                node.elts[i] = ast.Constant(value=randomize_primitive(elem.value))
        
        if len(node.elts) == 0:
            node.elts.append(ast.Constant(value=randomize_primitive(None)))

    elif isinstance(node, ast.Dict):
        # Randomly add a key-value pair
        if node.keys and random.random() < threshold and not nested:
            random_key = randomize_primitive(node.keys[0].value)
            random_value = randomize_primitive(node.values[0].value)
            node.keys.append(ast.Constant(value=random_key))
            node.values.append(ast.Constant(value=random_value))

        # Randomly delete a key-value pair
        if len(node.keys) > 1 and not nested and random.random() < threshold:
            idx = random.randint(0, len(node.keys) - 1)
            del node.keys[idx]
            del node.values[idx]

        # Recursive modification for keys and values
        for i, value in enumerate(node.values):
            if isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                randomize_output_structure(value, threshold, nested=True)
            elif isinstance(value, ast.Constant) and random.random() < threshold:
                node.values[i] = ast.Constant(value=randomize_primitive(value.value))
                
        if len(node.keys) == 0:
            node.keys.append(ast.Constant(value=randomize_primitive("")))
            node.values.append(ast.Constant(value=randomize_primitive("")))
    
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and not nested:
            node.operand = ast.Constant(value=randomize_primitive(-node.operand.value))

    elif isinstance(node, ast.Constant) and not nested:
        node.value = randomize_primitive(node.value)


def random_modify_output(original_output: str, follow_structure: bool = True, threshold: float = 0.5) -> str:
    """Modify one element in the original output structure."""
    if follow_structure:
        modifie_value = original_output
    else:
        modifie_value = repr(get_useless_value(type(ast.literal_eval(original_output)))[0])
          
    tree = ast.parse(modifie_value, mode='eval')
    randomize_output_structure(tree.body, threshold)
    return ast.unparse(tree.body)

"======================================================================================================"
def process_MDC(code: str, output: str = None, once: bool = False, p: int = 1) -> str:
    """
    Inserts misleading comments into the provided code.
        - code (str): The code snippet to modify.
        - output (str): The expected output value of the code.
        - once (bool): A flag indicating whether to insert comments only once.
        - p (int): The probability of inserting a comment.
    """
    tree = ast.parse(code)
    output = ast.literal_eval(output) if output is not None else None
    output_type = type(output) if output is not None else None
    
    lines = code.split("\n")
    
    class MisleadingCommentInserter(ast.NodeVisitor):
        def __init__(self, lines, output_type):
            self.lines = lines
            self.output_type = output_type
            self.initialized_variables = set()
            self.offset = 0

            self.contains = {
                "input": False,
                "return": False,
                "variable": False,
                "loop": False,
                "conditional": False,
                "operator": False,
                "operation": False
            }
        
        def should_insert(self, category):
            return not (once and self.contains[category]) and random.random() <= p
        
        def visit_FunctionDef(self, node):
            if self.should_insert("input"):
                comment = random.choice(INPUT_PARAMETERS_COMMENT_CANDIDATE)
                self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                self.contains["input"] = True
            self.generic_visit(node)
        
        def visit_Return(self, node):
            if self.should_insert("return"):
                useless_value = get_useless_value(self.output_type)[1]
                comment = random.choice(RETURN_STATEMENTS_COMMENT_CANDIDATE).format(useless_value=useless_value)
                self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                self.contains["return"] = True
            self.generic_visit(node)
        
        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.BinOp) and self.should_insert("operator"):
                    comment = random.choice(OPERATORS_COMMENT_CANDIDATE)
                    self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                    self.contains["operator"] = True
                elif var_name not in self.initialized_variables and self.should_insert("variable"):
                    self.initialized_variables.add(var_name)
                    comment = random.choice(VARIABLE_ASSIGNMENTS_COMMENT_CANDIDATE).format(variable=var_name)
                    self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                    self.contains["variable"] = True
            self.generic_visit(node)
        
        def visit_AugAssign(self, node):
            if isinstance(node.target, (ast.Name, ast.Subscript)) and self.should_insert("operator"):
                comment = random.choice(OPERATORS_COMMENT_CANDIDATE)
                self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                self.contains["operator"] = True
            self.generic_visit(node)
        
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                obj_name = node.func.value.id if isinstance(node.func.value, ast.Name) else "the object"
                if method_name in OPERATIONS_COMMENTS_CANDIDATE and self.should_insert("operation"):
                    comment = random.choice(OPERATIONS_COMMENTS_CANDIDATE[method_name]).format(name=obj_name)
                    self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                    self.contains["operation"] = True
            self.generic_visit(node)
        
        def visit_Loop(self, node):
            if self.should_insert("loop"):
                comment = random.choice(LOOP_STATEMENTS_COMMENT_CANDIDATE)
                if node.body:
                    first_body_line = node.body[0].lineno - 2 + self.offset
                    indent = " " * node.col_offset
                    self.lines.insert(first_body_line, f"{indent}# {comment}")
                    self.offset += 1
                    self.contains["loop"] = True
            self.generic_visit(node)
        
        def visit_For(self, node):
            self.visit_Loop(node)
        
        def visit_While(self, node):
            self.visit_Loop(node)
        
        def visit_If(self, node):
            if self.should_insert("conditional"):
                comment = random.choice(CONDITIONAL_STATEMENTS_COMMENT_CANDIDATE)
                self.lines[node.lineno - 1 + self.offset] += f"    # {comment}"
                self.contains["conditional"] = True
            self.generic_visit(node)
    
    inserter = MisleadingCommentInserter(lines, output_type)
    inserter.visit(tree)
    return "\n".join(lines)


def process_MPS(code: str, output: str = None) -> str:
    tree = ast.parse(code)
    output = ast.literal_eval(output) if output is not None else None
    output_type = type(output) if output is not None else None

    lines = code.split("\n")
    
    class MisleadingPrintInserter(ast.NodeVisitor):
        def __init__(self, lines, output_type):
            self.lines = lines
            self.output_type = output_type
            self.offset = 0
            self.parent_map = {}
            self.num_added_prints = 0

        def build_parent_map(self, node):
            for child in ast.iter_child_nodes(node):
                self.parent_map[child] = node
                self.build_parent_map(child)

        def insert_print(self, lineno, col_offset, comment):
            indent = " " * col_offset
            self.lines.insert(lineno, f'{indent}print("{comment}")')
            self.offset += 1
            self.num_added_prints += 1

        def visit_FunctionDef(self, node):
            comment = random.choice(INPUT_PARAMETERS_COMMENT_CANDIDATE)
            self.insert_print(node.lineno + self.offset, node.col_offset+4, comment)
            self.generic_visit(node)

        def visit_Return(self, node):
            useless_value = get_useless_value(self.output_type)[1]
            comment = random.choice(RETURN_STATEMENTS_COMMENT_CANDIDATE).format(useless_value=useless_value)
            self.insert_print(node.lineno - 1 + self.offset, node.col_offset, comment)
            self.generic_visit(node)

        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.BinOp):
                    comment = random.choice(OPERATORS_COMMENT_CANDIDATE)
                    self.insert_print(node.lineno - 1 + self.offset, node.col_offset, comment)
                else:
                    comment = random.choice(VARIABLE_ASSIGNMENTS_COMMENT_CANDIDATE).replace("{variable}", var_name)
                    self.insert_print(node.lineno - 1 + self.offset, node.col_offset, comment)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            comment = random.choice(OPERATORS_COMMENT_CANDIDATE)
            self.insert_print(node.lineno - 1 + self.offset, node.col_offset, comment)
            self.generic_visit(node)

        def visit_Call(self, node):
            line_text = self.lines[node.lineno - 1 + self.num_added_prints]
            leading_spaces = len(line_text) - len(line_text.lstrip(' '))
            col_offset = leading_spaces

            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                obj_name = (
                    node.func.value.id
                    if isinstance(node.func.value, ast.Name)
                    else "the object"
                )
                if method_name in OPERATIONS_COMMENTS_CANDIDATE:
                    comment = random.choice(OPERATIONS_COMMENTS_CANDIDATE[method_name]).replace("{name}", obj_name)
                    self.insert_print(node.lineno - 1 + self.offset, col_offset, comment)
            self.generic_visit(node)

        def visit_For(self, node):
            comment = random.choice(LOOP_STATEMENTS_COMMENT_CANDIDATE)
            self.insert_print(node.body[0].lineno - 2 + self.offset, node.col_offset, comment)
            self.generic_visit(node)

        def visit_While(self, node):
            comment = random.choice(LOOP_STATEMENTS_COMMENT_CANDIDATE)
            self.insert_print(node.body[0].lineno - 2 + self.offset, node.col_offset, comment)
            self.generic_visit(node)

        def visit_If(self, node):
            comment = random.choice(CONDITIONAL_STATEMENTS_COMMENT_CANDIDATE)
            self.insert_print(node.lineno + self.offset, node.col_offset+4, comment)
            self.generic_visit(node)

    inserter = MisleadingPrintInserter(lines, output_type)
    inserter.build_parent_map(tree)
    inserter.visit(tree)
    return ast.unparse(ast.parse("\n".join(lines)))
