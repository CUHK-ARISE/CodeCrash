import ast
from typing import Dict, List, Set, Union

from perturbations.structural.utils import parse_code, is_reserved


# ---------- Variable collection ----------
def _collect_variables(node: ast.AST) -> Set[str]:
    """
    Collects variable names assigned in the given AST node.
    
    Args:
        node (ast.AST): The AST node to extract variable names from.
    """
    vars = set()

    # Handle assignments (e.g., a = 1 or a, b = 1, 2)
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                vars.add(t.id)
            elif isinstance(t, ast.Tuple):
                for e in t.elts:
                    if isinstance(e, ast.Name):
                        vars.add(e.id)
    
    # Handle for-loop variables (e.g., for i in range(10))
    elif isinstance(node, ast.For):
        tgt = node.target
        if isinstance(tgt, ast.Name):
            vars.add(tgt.id)
        elif isinstance(tgt, ast.Tuple):
            for e in tgt.elts:
                if isinstance(e, ast.Name):
                    vars.add(e.id)
    
    # Handle with-statement variables (e.g., with open() as f)
    elif isinstance(node, ast.With):
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                vars.add(item.optional_vars.id)
    
    # Handle walrus operator (e.g., if(a := 10) > 5)
    elif isinstance(node, ast.NamedExpr):
        if isinstance(node.target, ast.Name):
            vars.add(node.target.id)
    
    # Handle comprehensions (e.g., [x for x in range(10)])
    elif isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                vars.add(gen.target.id)

    # Recursively process children, but skip descending into nested functions.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        vars.update(_collect_variables(child))

    return vars


def _collect_local_variables(funct_node: ast.FunctionDef) -> Set[str]:
    """
    Collects local variable names in the given function node.
    
    Args:
        funct_node (ast.FunctionDef): The function definition node to extract local variables from.
    """
    vars: Set[str] = set()
    for child in funct_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        vars.update(_collect_variables(child))
    return vars


def _collect_parameters(funct_node: ast.FunctionDef) -> Set[str]:
    """
    Collects parameter names in the given function node.
    
    Args:
        funct_node (ast.FunctionDef): The function definition node to extract parameters from.
    """
    params: Set[str] = set()
    
    # Handle regular parameters (e.g., def f(a, b))
    for a in funct_node.args.args:
        params.add(a.arg)

    # Handle default parameters (e.g., def f(a=1, b=2))
    for d in funct_node.args.defaults:
        if isinstance(d, ast.Name):
            params.add(d.id)

    # Handle *args (e.g., def f(*args))
    for d in funct_node.args.defaults:
        if isinstance(d, ast.Name):
            params.add(d.id)
    
    # Handle keyword-only parameters (e.g., def f(*, a, b))
    for a in funct_node.args.kwonlyargs:
        params.add(a.arg)
    
    # Handle default values for keyword-only parameters (e.g., def f(*, a=1, b=2))
    for d in funct_node.args.kw_defaults:
        if isinstance(d, ast.Name):
            params.add(d.id)
    
    return params


# ---------- Function discovery ----------
def get_function_structure(code_or_tree: Union[str, ast.AST]) -> List[Dict]:
    """
    Retrieves a list of dictionaries representing all functions in the given code or AST.
    
    Args:
        code_or_tree (str | ast.AST): The Python source code or AST to parse.
    """
    tree = parse_code(code_or_tree) if isinstance(code_or_tree, str) else code_or_tree

    def extract(node: ast.AST) -> List[Dict[str, Union[str, list]]]:
        out: List[Dict] = []
        body = getattr(node, "body", [])
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append({
                    "name": child.name,
                    "parameters": _collect_parameters(child),
                    "local_variables": _collect_local_variables(child),
                    "node": child,
                    "nested": extract(child),
                })
            elif isinstance(child, ast.ClassDef):
                out.extend(extract(child))
        return out

    return extract(tree)


def get_function_names(code_or_tree: Union[str, ast.AST]) -> Set[str]:
    """
    Retrieves a set of all function names in the given code or AST.
    
    Args:
        code_or_tree (str | ast.AST): The Python source code or AST to parse.
    """
    struct = get_function_structure(code_or_tree)
    names: Set[str] = set()

    def rec(items: List[Dict]):
        for it in items:
            names.add(it["name"])
            rec(it["nested"])

    rec(struct)
    return names


def get_parameters(code_or_tree: Union[str, ast.AST], target_funct: Union[str, List[str], None] = None, nested: bool = True) -> List[str]:
    """
    Retrieves a list of parameter names in the given code or AST.
    
    Args:
        code_or_tree (str | ast.AST): The Python source code or AST to parse.
        target_funct (str | List[str] | None): The target function name(s) to filter by. If None, includes all functions.
        nested (bool): Whether to include parameters from nested functions.
    """
    
    struct = get_function_structure(code_or_tree)
    if isinstance(target_funct, str):
        target_funct = [target_funct]

    params: Set[str] = set()

    def rec(items: List[Dict]):
        for it in items:
            if target_funct is None or it["name"] in target_funct:
                params.update(it["parameters"])
            if nested and it["nested"]:
                rec(it["nested"])

    rec(struct)
    return [p for p in params if not is_reserved(p)]


def get_local_variables(code_or_tree: Union[str, ast.AST], target_funct: Union[str, List[str], None] = None, nested: bool = True) -> List[str]:
    """
    Retrieves a list of local variable names in the given code or AST.

    Args:
        code_or_tree (str | ast.AST): The Python source code or AST to parse.
        target_funct (str | List[str] | None): The target function name(s) to filter by. If None, includes all functions.
        nested (bool): Whether to include variables from nested functions.
    """
    struct = get_function_structure(code_or_tree)
    if isinstance(target_funct, str):
        target_funct = [target_funct]

    locals_: Set[str] = set()

    def rec(items: List[Dict]):
        for it in items:
            if target_funct is None or it["name"] in target_funct:
                locals_.update(it["local_variables"])
            if nested and it["nested"]:
                rec(it["nested"])

    rec(struct)
    return [v for v in locals_ if not is_reserved(v)]


def get_variables(code_or_tree: Union[str, ast.AST]) -> List[str]:
    """
    Retrieves a list of all variable (including global variables) names in the given code or AST.
    
    Args:
        code_or_tree (str | ast.AST): The Python source code or AST to parse
    """
    tree = parse_code(code_or_tree) if isinstance(code_or_tree, str) else code_or_tree
    names: Set[str] = set()
    for n in ast.walk(tree):
        names.update(_collect_variables(n))
    return [v for v in names if not is_reserved(v)]