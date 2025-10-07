import ast
from typing import Dict, Tuple, Union, List

from perturbations.structural.renaming import FunctionRenamer, ScopeAwareIdentifierRenamer, get_function_names, get_parameters, get_variables
from perturbations.structural.reformatting import ConditionReformatter
from perturbations.structural.garbage import GarbageCodeInsertor
from perturbations.structural.utils import parse_code, unparse_code


def add_parent_links(node: ast.AST, parent: ast.AST | None = None):
    for child in ast.iter_child_nodes(node):
        setattr(child, "parent", node)
        add_parent_links(child, node)


def rename_function_definitions(code: str, target_funct_name: str = None, expressions: List[str] = []) -> Tuple[str, List[str]]:
    """
    Rename only top-level function definitions in the given code.

    Args:
        code (str): Python source code.
        target_funct_name (str, optional): The main target function to rename to 'f'.
        expressions (List[str], optional): Expressions (e.g., test calls) to rename accordingly.
    """
    def _rename_in_call_expr(expr: str, mapping_ref: Dict[str, str]) -> str:
        tree = ast.parse(expr, mode="exec")
        tree = FunctionRenamer(mapping_ref).visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).strip()
    
    tree = parse_code(code)
    funct_names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funct_names.append(node.name)
    
    # Build renaming mapping
    mapping_ref: Dict[str, str] = {}
    if target_funct_name is not None:
        if target_funct_name not in funct_names:
            raise ValueError(f"Target function '{target_funct_name}' not found among top-level functions.")
        other_funcs = [n for n in funct_names if n != target_funct_name]
        mapping_ref[target_funct_name] = "f"
        for i, name in enumerate(other_funcs, start=1):
            mapping_ref[name] = f"f{i}"
    else:
        for i, name in enumerate(funct_names, start=1):
            mapping_ref[name] = f"f{i}"
    
    # Apply renaming to main code
    tree = FunctionRenamer(mapping_ref).visit(tree)
    ast.fix_missing_locations(tree)
    new_code = unparse_code(tree).strip()
    
    # Apply renaming to expressions
    new_expressions = []
    for expr in expressions:
        new_expressions.append(_rename_in_call_expr(expr, mapping_ref))

    return new_code, new_expressions


def rename_variables(code: str, expressions: List[str] = []) -> Tuple[str, List[str]]:
    """
    Rename all variables and parameters in the given code to Var_1, Var_2, ...
    
    Args:
        code (str): The Python source code to process.
        expressions (List[str], optional): The execution expressions to update.
    """
    def _rename_in_call_expr(expr: str, mapping_ref: Dict[str, str]) -> str:
        """
        Apply renaming to variable names in the expression, skipping function names.
        """
        tree = ast.parse(expr, mode="exec")
        add_parent_links(tree)
        tree = ScopeAwareIdentifierRenamer(mapping_ref).visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).strip()
    
    tree = parse_code(code)
    params = get_parameters(tree)
    vars_ = get_variables(tree)
    identities = sorted(set(params + vars_))
    mapping_ref: Dict[str, str] = {
        name: f"Var_{i+1}" for i, name in enumerate(identities, start=1)
    }
    
    # Apply renaming to main code
    tree = ScopeAwareIdentifierRenamer(mapping_ref).visit(tree)
    ast.fix_missing_locations(tree)
    new_code = unparse_code(tree).strip()

    # Avoid renaming function names in expressions
    func_names = {
        node.name for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef)
    }
    safe_mapping = {
        k: v for k, v in mapping_ref.items()
        if k not in func_names
    }

    # Apply renaming to expressions
    new_expressions = []
    for expr in expressions:
        new_expressions.append(_rename_in_call_expr(expr, safe_mapping))

    return new_code, new_expressions


def process_renaming_entities(code: str, target_funct_name: str = None, expressions: List[str] = []) -> Tuple[str, Union[str, None], List[str]]:
    """
    Perturbation Flag: REN (Renaming Entities)
    Rename function definitions and variables in the given code.
    
    Args:
        code (str): The Python source code to process.
        
        target_funct_name (str, optional): The function name to rename to 'f'.
            - If provided, that function becomes 'f' and others are 'f1', 'f2', ...
            - If None, all functions are renamed sequentially as 'f1', 'f2', ...
        
        expressions (List[str], optional): The execution expressions to update.
    """
    new_code, new_expressions = rename_function_definitions(code, target_funct_name, expressions)
    new_code, new_expressions = rename_variables(new_code, new_expressions)
    new_funct_name = "f" if target_funct_name is not None else None
    return new_code, new_funct_name, new_expressions


def process_reformatting_conditions(code: str) -> str:
    """
    Perturbation Flag: RTF (Reformatting Conditions)
    Reformat conditions in if and while statements using randomized templates.
    
    Args:
        code (str): The Python source code to process.
    """
    tree = parse_code(code)
    new_tree = ConditionReformatter(code).visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = unparse_code(new_tree).strip()
    return new_code


def process_inserting_garbage_code(code: str) -> str:
    """
    Perturbation Flag: GBC (Inserting Garbage Code)
    Insert garbage code into the given program:
        1. Inserting redundant global assignments at the module top.
        2. Inserting inexplicable code segments into random functions at random valid spots.
        3. Inserting death loops functions at random valid spots.
    
    Args:
        code (str): The Python source code to process.
    """
    tree = parse_code(code)
    new_tree = GarbageCodeInsertor(code).visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = unparse_code(new_tree).strip()
    return new_code


def process_psc_all(code: str, target_funct_name: str = None, expressions: List[str] = []) -> Tuple[str, Union[str, None], List[str]]:
    """
    Perturbation Flag: PSC-ALL (Aggregated Program Structure-Consistent [PSC] Perturbations)
    Apply all three PSC perturbations to the given code:
        1. Rename function definitions and variables (REN).
        2. Reformat conditions in if/while statements (RTF).
        3. Insert garbage code (GBC).
    
    Args:
        code (str): The Python source code to process.
        target_funct_name (str, optional): The function name to rename.
        call_expr (str, optional): The execution expression to update.
    """
    new_code = process_reformatting_conditions(code)
    new_code = process_inserting_garbage_code(new_code)
    new_code, new_funct_name, new_expressions = process_renaming_entities(new_code, target_funct_name, expressions)
    return new_code, new_funct_name, new_expressions