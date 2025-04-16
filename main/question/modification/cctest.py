import ast
import astor
import random
import math
from redbaron import RedBaron
from typing import List, Dict, Any, Union, Set, Tuple


def add_parent_links(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def get_enclosing_function_names(node: ast.AST) -> list:
    """ Retrieves the names of all enclosing functions for the given AST node."""
    function_names = []
    while node:
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
        node = getattr(node, 'parent', None)
    return function_names[::-1]


def get_function_name(node: ast.AST) -> str:
    """ Retrieves the name of the function that the given AST node belongs to. """
    while node:
        if isinstance(node, ast.FunctionDef):
            return node.name
        node = getattr(node, 'parent', None)  # Traverse upwards in the AST tree
    return None


def collect_parameters(node: ast.AST) -> Set[str]:
    """ Collects parameters from a function definition node. """
    parameters = set()
    
    # Handle regular arguments (e.g., def func(a))
    for arg in node.args.args:
        parameters.add(arg.arg)

    # Handle default values (e.g., def func(a=1))
    for default in node.args.defaults:
        if isinstance(default, ast.Name):
            parameters.add(default.id)

    # Handle keyword-only arguments (e.g., def func(*, b))
    for arg in node.args.kwonlyargs:
        parameters.add(arg.arg)

    # Handle keyword-only defaults (e.g., def func(*, b=2))
    for default in node.args.kw_defaults:
        if isinstance(default, ast.Name):
            parameters.add(default.id)
    
    return parameters


def collect_variables(node: ast.AST) -> Set[str]:
    """
    Finds all variable names defined in any node of the AST subtree rooted at `node`,
    skipping nested function and async function definitions (but processing lambda expressions).
    """
    variables = set()
    
    # Process the current node if it defines variables.
    if isinstance(node, ast.Assign):
        # Handle assignment statements (e.g., a = 1, or a, b = 1, 2)
        for target in node.targets:
            if isinstance(target, ast.Name):
                variables.add(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        variables.add(element.id)
    elif isinstance(node, ast.For):
        # Handle for-loop target variables (e.g., for i in range(10))
        if isinstance(node.target, ast.Name):
            variables.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for element in node.target.elts:
                if isinstance(element, ast.Name):
                    variables.add(element.id)
    elif isinstance(node, ast.With):
        # Handle with-statement variables (e.g., with open('file') as f)
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                variables.add(item.optional_vars.id)
    elif isinstance(node, ast.NamedExpr):
        # Handle the walrus operator (e.g., if (a := 10) > 5)
        if isinstance(node.target, ast.Name):
            variables.add(node.target.id)
    elif isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        # Handle comprehensions (e.g., [x for x in range(10)] or (x for x in range(10)))
        for generator in node.generators:
            if isinstance(generator.target, ast.Name):
                variables.add(generator.target.id)
    
    # Recursively process children, but skip descending into nested functions.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip nested function definitions
            continue
        # Lambdas are not skipped
        variables.update(collect_variables(child))
    
    return variables

def collect_local_variables(node: ast.FunctionDef) -> Set[str]:
    """
    Collects all local variables within a function definition node,
    excluding those defined in nested function (or async function) bodies, 
    but including variables in lambda expressions and comprehensions.
    """
    variables = set()
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        variables.update(collect_variables(child))
    return variables


def get_function_structure(code: str) -> List[Dict]:
    """
    Retrieves a list of dictionaries representing all functions in the given code.
        code (str): The Python source code to parse.
    """
    tree = ast.parse(code)

    def extract_functions(node: ast.AST) -> List[Dict[str, Union[str, list]]]:
        """
        Recursively extracts functions (including lambdas) from any node in the AST.
        """
        functions = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                functions.append({
                    "name": child.name,
                    "parameters": collect_parameters(child),
                    "local_variables": collect_local_variables(child),
                    "node": child,
                    "nested": extract_functions(child)
                })
        return functions
    
    function_structure = extract_functions(tree)
    return function_structure


def get_function_names(code: str) -> Set[str]:
    """Recursively gather all function names (including nested) into a list."""
    function_structure = get_function_structure(code)
    function_names = set()
    
    def collect_functions(structure: dict):
        for item in structure:
            function_names.add(item['name'])
            collect_functions(item['nested'])
    
    collect_functions(function_structure)
    return function_names


def get_parameters(
    code: str,
    target_funct: Union[str, List[str], None] = None,
    nested: bool = True
) -> List[str]:
    """
    Retrieves the parameters of the target function(s) in the given code.
        code (str): The Python source code to parse.
        target_funct (str, list, None): The target function(s) to extract parameters from.
        nested (bool): Whether to include nested functions.
    """
    parameters = set()
    function_structure = get_function_structure(code)
    
    if isinstance(target_funct, str):
        target_funct = [target_funct]

    def extract_parameters(functions):
        for func in functions:
            if target_funct is None or func["name"] in target_funct:
                parameters.update(func["parameters"])
            if nested and func["nested"]:
                extract_parameters(func["nested"])

    extract_parameters(function_structure)
    return list(parameters)


def get_local_variables(
    code: str,
    target_funct: Union[str, List[str], None] = None,
    nested: bool = True
) -> List[str]:
    """
    Retrieves the local variables of the target function(s) in the given code.
        code (str): The Python source code to parse.
        target_funct (str, list, None): The target function(s) to extract local variables from.
        nested (bool): Whether to include nested functions.
    """
    local_variables = set()
    function_structure = get_function_structure(code)
    
    if isinstance(target_funct, str):
        target_funct = [target_funct]

    def extract_local_variables(functions):
        for func in functions:
            if target_funct is None or func["name"] in target_funct:
                local_variables.update(func["local_variables"])
            if nested and func["nested"]:
                extract_local_variables(func["nested"])

    extract_local_variables(function_structure)
    return list(local_variables)


def get_variables(code) -> List[str]:
    """
    Finds all local variables in the code using the Python AST module.
    """
    variables = set()
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        variables.update(collect_variables(node))
    
    return list(variables)


"===================================================================================================="
"""
Flag: REF - Rename Function
"""
def rename_one_function_name(code: str, old_name: str, new_name: str) -> str:
    """
    Renames a single function (or function variable) from `old_name` to `new_name`:
      - in its definition
      - in direct calls like old_name(...)
      - in attribute calls like old_name.cache_clear(), i.e. old_name is the 'value' in an ast.Attribute.
    Returns the transformed code.
    """
    class SingleFunctionRenamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            if node.name == old_name:
                node.name = new_name
            self.generic_visit(node)
            return node

        def visit_Call(self, node: ast.Call):
            """
            node.func can be:
              - ast.Name (e.g., old_name(...))
              - ast.Attribute (e.g., old_name.cache_clear(...))
            We rename the underlying Name if it matches old_name.
            """
            # First visit children (in case there are calls within arguments)
            self.generic_visit(node)

            if isinstance(node.func, ast.Name):
                if node.func.id == old_name:
                    node.func.id = new_name

            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == old_name:
                    node.func.value.id = new_name

            return node

        def visit_Attribute(self, node: ast.Attribute):
            """
            Even if it's not a direct call, we might have references like:
                old_name.something  (e.g. old_name.cache_clear)
            If we want to rename the base Name, we handle that here too.
            """
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == old_name:
                node.value.id = new_name
            return node
        
        def visit_Name(self, node: ast.Name):
            """
            Catch-all for any usage of `old_name` as a simple Name in the code
            (e.g., passing it as an argument, referencing it in an expression, etc.).
            """
            if node.id == old_name:
                node.id = new_name
            return node

    tree = ast.parse(code)
    transformer = SingleFunctionRenamer()
    new_tree = transformer.visit(tree)
    updated_code = ast.unparse(new_tree)
    return updated_code



def process_REF(code: str, target_function: str, function_call: str) -> Tuple[str, str, str]:
    """
    Rename the target function to `f` and all other functions to `f1`, `f2`, ..., `fn`.
    """
    function_names = get_function_names(code)
    function_names.remove(target_function)

    new_code = code
    new_function_call = function_call
    
    # Rename target function
    new_target_name = "f"
    new_code = rename_one_function_name(new_code, target_function, new_target_name)
    new_function_call = rename_one_function_name(new_function_call, target_function, new_target_name)
    
    # Rename other functions
    for i, old_name in enumerate(sorted(function_names), start=1):
        new_name = f"f{i}"
        new_code = rename_one_function_name(new_code, old_name, new_name)
        new_function_call = rename_one_function_name(new_function_call, old_name, new_name)
    return new_code.strip(), new_target_name, new_function_call.strip()
        

"===================================================================================================="
"""
Flag: RPV - Rename Parameters and Variables (Regulate)
Example:
    - f(a, b) -> f(Para_1, Para_2)
    - c = 1 -> Para_3 = 1
"""
def rename_one_variable(code: str, old_name: str, new_name: str) -> str:
    """ Renames a variable or parameter in the code using AST parsing. """
    tree = ast.parse(code)

    def rename_node(node):
        if isinstance(node, ast.arg) and node.arg == old_name:
            node.arg = new_name
        elif isinstance(node, ast.Name) and node.id == old_name:
            node.id = new_name
        for child in ast.iter_child_nodes(node):
            rename_node(child)

    rename_node(tree)
    return ast.unparse(tree)


def rename_expression(function_name: str, expression: str, old_name: str, new_name: str) -> str:
    class RenameVisitor(ast.NodeTransformer):
        def __init__(self):
            self.target_funct = False

        def visit_Call(self, node):
            """Check if the function call matches the target function."""
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                self.target_funct = True
                node.args = [self.visit(arg) for arg in node.args]
                node.keywords = [self.visit(kw) for kw in node.keywords]
                self.target_funct = False
            return node

        def visit_Name(self, node):
            """Rename only if inside the target function's positional arguments."""
            if self.target_funct and node.id == old_name:
                node.id = new_name
            return node

        def visit_keyword(self, node):
            """Rename keyword arguments only if inside the target function."""
            if self.target_funct and node.arg == old_name:
                node.arg = new_name
            return node

    tree = ast.parse(expression, mode='eval')
    tree = RenameVisitor().visit(tree)
    ast.fix_missing_locations(tree)
    
    new_expression = ast.unparse(tree).strip()
    
    if new_expression.startswith("(") and new_expression.endswith(")"):
        new_expression = new_expression[1:-1].strip()
    
    return new_expression.strip()


def process_RPV(code: str, function_name: str, expression: str) -> Tuple[str, str]:
    expression = expression.replace("assert ", "").strip()
    
    parameters = get_parameters(code)
    lvariables = get_local_variables(code)
    variables = parameters + lvariables
        
    for idx, var in enumerate(variables):
        new_name = f"Var_{idx+1}"
        code = rename_one_variable(code, var, new_name)
        if var in parameters:
            expression = rename_expression(function_name, expression, var, new_name)
    
    return code.strip(), expression.strip()


"===================================================================================================="
"""
Flag: RTF - Replace Boolean Expression
Example (p is a parameter):
    - if a < b: -> if (a < b) == (p == p):
    - if True: -> if True != (p != p):
Checks whether the condition is valid for modification:
    1. Boolean literals (True, False)
    2. Comparisons (e.g., a < b, a == b)
    -  Ignore variables like 'if a:' because they may not be boolean type
"""

# Templates for obfuscating boolean expressions.
gen_template = [
    "not not ({cond})",
    "({cond}) or False",
    "({cond}) and True",
    "_ := ({cond},)[0]",
]

comp_templates = [
    "_ := ({cond},)[0]",
    "(lambda: {cond})()",
    "({cond}) == ({para} == {para})",
    "({cond}) != ({para} != {para})",
    "bool(-~({cond})) == ({cond})",
    "bool(int({cond}))",
]

const_templates = [
    "_ := ({cond},)[0]",
    "(lambda: {cond})()",
    "({cond}) == ({para} == {para})",
    "({cond}) != ({para} != {para})",
    "bool(-~({cond})) == ({cond})",
    "eval(str({cond}))",
]

def process_RTF(code: str) -> str:
    
    def get_template(node: ast.AST) -> List[str]:
        if isinstance(node, ast.Compare):
            return comp_templates
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return const_templates
        return gen_template
    
    def modify_condition(condition: str, parameter: str, template: str) -> str:
        return template.format(cond=condition, para=parameter)
    
    def apply_replacements(code: str, replacements: List[Tuple[ast.AST, Set[str], List[str]]]) -> str:
        """
        Apply replacements based on AST node positions, sorted in reverse 
        so we don't mess up offsets for subsequent replacements.
        """
        replacements = sorted(replacements, key=lambda x: x[0].end_col_offset, reverse=True)
        code_lines = code.splitlines()
        
        for node, parameters, templates in replacements:
            start_line = node.lineno - 1
            start_col = node.col_offset
            end_col = node.end_col_offset

            original_condition = code_lines[start_line][start_col:end_col]

            has_colon = False
            if original_condition.endswith(':'):
                has_colon = True
                original_condition = original_condition[:-1]

            new_condition = modify_condition(
                original_condition,
                random.choice(parameters),
                random.choice(templates)
            )

            replacement = new_condition + (':' if has_colon else '')

            code_lines[start_line] = (
                code_lines[start_line][:start_col]
                + replacement
                + code_lines[start_line][end_col:]
            )

        return "\n".join(code_lines)
    
    tree = ast.parse(code)
    add_parent_links(tree)
    
    replacements = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            templates = get_template(node.test)
            enclosing_functions = get_enclosing_function_names(node)
            parameters = get_parameters(code, enclosing_functions)
            parameters = parameters + ["True", "False"]
            replacements.append((node.test, parameters, templates))
    
    modified_code = apply_replacements(code, replacements)
    return modified_code



"===================================================================================================="
"""
Flag: IRR - Instruction Replacement
Example:
    - a = a + 1 -> a += 1
    - a *= 1 -> a = a * 1
"""
def process_IRR(code: str) -> str:
    
    class AugAssignTransformer(ast.NodeTransformer):
        # Handles 'a = a + b' to 'a += b'
        def visit_Assign(self, node):
            if (
                isinstance(node.targets[0], ast.Name) and  # Left-hand side must be a variable
                isinstance(node.value, ast.BinOp) and  # Right-hand side must be a binary operation
                isinstance(node.value.left, ast.Name) and  # Left side of operation must be a variable
                node.targets[0].id == node.value.left.id  # Variables must match
            ):
                # Replace with Augmented Assignment
                return ast.AugAssign(
                    target=node.targets[0],
                    op=node.value.op,
                    value=node.value.right
                )
            return node

        # Handles 'a += b' to 'a = a + b'
        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name):
                return ast.Assign(
                    targets=[node.target],
                    value=ast.BinOp(
                        left=node.target,
                        op=node.op,
                        right=node.value
                    )
                )
            return node
    
    tree = ast.parse(code)
    transformer = AugAssignTransformer()
    transformed_tree = transformer.visit(tree)
    ast.fix_missing_locations(transformed_tree)
    return ast.unparse(transformed_tree)


"===================================================================================================="
"""
Flag: GRA - Garbage Code Insertion
Example:
    - If (p != p):
    - While False
    - For i in range(0):
"""

import ast
import random
import astor

def get_garbage_templates():
    """
    Provides a list of conditions and code formats for generating garbage code.

    Returns:
        tuple: A list of conditions and formats for garbage code.
    """
    false_cond = [
        "False",
        "None",
        "0",
        "''",
        "{parameter} != {parameter}",
        "not {parameter} == {parameter}",
        "print({parameter})",
    ]

    inexplicable_codes = [
        "if {false_cond}: {new_var} = {parameter}",
        "while {false_cond}: {new_var} = {parameter}",
        "for i in range(0): {new_var} = {parameter}",
        "{new_var} = {parameter} if {false_cond} else {parameter}",
    ]
    
    garbage_death_loops = [
        "def funct1():\n    funct2()\n\ndef funct2():\n    funct1()",
        "def funct3():\n    def funct4():\n        funct3()\n    funct4()",
        "def funct5():\n    i = 1\n    while True:\n        i+=1",
        "def funct6():\n    for i in iter(int, 1):\n        i+=1",
        "def funct7():\n    try:\n        funct7()\n    except:\n        funct7()",
        "def funct8():\n    items = [0]\n    for x in items:\n        items.append(x + 1)",
        "def funct9():\n    for _ in iter(lambda: True, False):\n        pass",
    ]
    
    selected_inexplicable_codes = [
        code.format(false_cond=random.choice(false_cond), parameter="{parameter}", new_var=f"TempVar{i}")
        for i, code in enumerate(random.sample(inexplicable_codes, 3))
    ]
    
    return selected_inexplicable_codes + random.sample(garbage_death_loops, 2)


def collect_insertion_points_in_body(body):
    """
    Recursively collect all valid insertion points in a list of statements, 
    stopping in a block after encountering a Return statement.
    """
    insertion_points = []
    i = 0

    while i <= len(body):
        insertion_points.append((body, i))
        
        if i == len(body):
            break
        
        stmt = body[i]
        if isinstance(stmt, ast.Return):
            break
        
        # --- If the current statement has nested blocks, recurse into them ---
        if isinstance(stmt, ast.If):
            insertion_points.extend(collect_insertion_points_in_body(stmt.body))
            insertion_points.extend(collect_insertion_points_in_body(stmt.orelse))
        
        elif isinstance(stmt, ast.For):
            insertion_points.extend(collect_insertion_points_in_body(stmt.body))
            insertion_points.extend(collect_insertion_points_in_body(stmt.orelse))
        
        elif isinstance(stmt, ast.While):
            insertion_points.extend(collect_insertion_points_in_body(stmt.body))
            insertion_points.extend(collect_insertion_points_in_body(stmt.orelse))
        
        elif isinstance(stmt, ast.With):
            insertion_points.extend(collect_insertion_points_in_body(stmt.body))
        
        elif isinstance(stmt, ast.FunctionDef):
            insertion_points.extend(collect_insertion_points_in_body(stmt.body))
        
        elif isinstance(stmt, ast.ClassDef):
            insertion_points.extend(collect_insertion_points_in_body(stmt.body))
                
        i += 1
    
    return insertion_points


def insert_code_to_node_randomly(node: ast.FunctionDef, code_to_insert: str) -> ast.FunctionDef:
    """
    Inserts code (given as a string) into a random valid location within the provided
    ast.FunctionDef node or its nested blocks (ifs, loops, nested defs, etc.), ensuring
    the new code is never placed after a return in the same block.
    """
    new_statements = ast.parse(code_to_insert).body
    
    insertion_points = collect_insertion_points_in_body(node.body)
    if not insertion_points:
        return node
    
    target_body, insert_index = random.choice(insertion_points)
    
    target_body[insert_index:insert_index] = new_statements
    
    return node


def process_GRA(code: str) -> str:
    """
    Insert each garbage code format into a randomly selected function at a random line.
    """
    garbage_templates = get_garbage_templates()
    random.shuffle(garbage_templates)
    tree = ast.parse(code)
    add_parent_links(tree)

    get_outer_functions = get_function_structure(code)
    function_names = [node["name"] for node in get_outer_functions]
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in function_names]

    for idx, gc in enumerate(garbage_templates):
        target_function = random.choice(functions)
        enclosing_functions = get_enclosing_function_names(target_function)
        parameters = get_parameters(code, enclosing_functions)
        
        if not parameters:
            parameters = [random.randint(1, 100) for _ in range(10)]
        
        garbage_code = gc.format(parameter=random.choice(parameters))
        insert_code_to_node_randomly(target_function, garbage_code)
    return ast.unparse(tree).strip()

"===================================================================================================="
"""
Flag: GGV - Garbage Repeated  Naming Global Variable
"""
def process_GGV(code: str) -> str:
    candidate_variables  = get_parameters(code)
    
    if len(candidate_variables) == 0:
        candidate_variables += get_local_variables(code)

    chosen_vars = random.sample(candidate_variables, math.ceil(len(candidate_variables) * 0.5))

    assignment_lines = []
    for var in chosen_vars:
        value = random.randint(1, 100)
        assignment_lines.append(f"{var} = {value}")

    lines = code.split('\n')
    new_lines = assignment_lines + [""] + lines
    new_code = "\n".join(new_lines)
    return ast.unparse(ast.parse(new_code)).strip()
