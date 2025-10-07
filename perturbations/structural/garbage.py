import ast
import random
import math
from typing import List, Tuple, Dict

from perturbations.structural.renaming import get_function_structure, get_parameters, get_local_variables


# ---------- Templates ----------
def get_garbage_templates():
    FALSE_CONDITIONS = [
        "False",
        "None",
        "0",
        "''",
        "{parameter} != {parameter}",
        "not {parameter} == {parameter}",
        "print({parameter})",
    ]
    
    INEXPLICABLE_CODE_TEMPLATES = [
        "if {false_cond}: {new_var} = {parameter}",
        "while {false_cond}: {new_var} = {parameter}",
        "for i in range(0): {new_var} = {parameter}",
        "{new_var} = {parameter} if {false_cond} else {parameter}",
    ]
    
    GARBAGE_DEATH_LOOP_TEMPLATES = [
        "def funct1():\n    funct2()\n\ndef funct2():\n    funct1()",
        "def funct3():\n    def funct4():\n        funct3()\n    funct4()",
        "def funct5():\n    i = 1\n    while True:\n        i+=1",
        "def funct6():\n    for i in iter(int, 1):\n        i+=1",
        "def funct7():\n    try:\n        funct7()\n    except:\n        funct7()",
        "def funct8():\n    items = [0]\n    for x in items:\n        items.append(x + 1)",
        "def funct9():\n    for _ in iter(lambda: True, False):\n        pass",
    ]
    
    selected_templates = [
        template.format(
            false_cond=random.choice(FALSE_CONDITIONS),
            parameter="{parameter}",
            new_var=f"TempVar{i}"
        )
        for i, template in enumerate(random.sample(INEXPLICABLE_CODE_TEMPLATES, 3))
    ]
    return selected_templates + random.sample(GARBAGE_DEATH_LOOP_TEMPLATES, 2)


class GarbageCodeInsertor(ast.NodeTransformer):
    """
    Insert garbage code into a program:
        1. Insert redundant global assignments at the module top.
        2. Insert inexplicable code segments into random functions at random valid spots.
        3. Insert death loops functions at random valid spots.
    """
    
    def __init__(self, code: str):
        super().__init__()
        self.original_code = code
        self._funct_params_by_id: Dict[int, List[str]] = {}
    
    
    # ---------- Helpers ----------
    def _build_funct_params_by_id(self, tree: ast.AST) -> None:
        struct = get_function_structure(tree)
        by_id: Dict[int, List[str]] = {}
        
        def rec(items):
            for item in items:
                fn_node = item["node"]
                params = list(item["parameters"])
                by_id[id(fn_node)] = params
                rec(item["nested"])
        rec(struct)
        self._funct_params_by_id = by_id
    
    def _collect_insertion_points(self, body: List[ast.stmt]) -> List[Tuple[List[ast.stmt], int]]:
        """
        Recursively collect all valid insertion points in a list of statements, stopping in a block after encountering a Return statement.
        """
        insertion_pts = []
        i = 0
        while i <= len(body):
            insertion_pts.append((body, i))
            if i == len(body):
                break
            
            stmt = body[i]
            if isinstance(stmt, ast.Return):
                break
            
            if isinstance(stmt, (ast.If, ast.For, ast.While)):
                insertion_pts.extend(self._collect_insertion_points(stmt.body))
                insertion_pts.extend(self._collect_insertion_points(stmt.orelse))
            elif isinstance(stmt, (ast.With, ast.FunctionDef, ast.ClassDef)):
                insertion_pts.extend(self._collect_insertion_points(stmt.body))
            i += 1
        return insertion_pts
    
    def _insert_code_to_function_randomly(self, funct: ast.FunctionDef, code_to_insert: str) -> None:
        """
        Insert the given code snippet into a random valid position within the function body.
        """
        new_stmts = ast.parse(code_to_insert).body
        insertion_pts = self._collect_insertion_points(funct.body)
        if not insertion_pts:
            return
        target_body, idx = random.choice(insertion_pts)
        target_body[idx:idx] = new_stmts
    
    def _insert_global_assignments(self, module: ast.Module, tree_for_analysis: ast.AST) -> None:
        """
        Insert redundant global assignments at the top of the module.
        """
        # Find parameters or local variables to use (if parameters not found, use local variables)
        params = get_parameters(tree_for_analysis)
        if not params:
            params += get_local_variables(tree_for_analysis)
        if not params:
            return
        
        #  Randomly choose half of them to create assignments
        k = max(1, math.ceil(len(params) * 0.5))
        chosen = random.sample(params, k)
        assigns_src = "\n".join(f"{name} = {random.randint(1, 100)}" for name in chosen)
        assign_nodes = ast.parse(assigns_src).body
        
        # Insert after docstring if present
        insert_at = 0
        if module.body and isinstance(module.body[0], ast.Expr) and isinstance(getattr(module.body[0], "value", None), ast.Constant) and isinstance(module.body[0].value.value, str):
            insert_at = 1
        module.body[insert_at:insert_at] = assign_nodes
    
    def _choose_parameter_for_function(self, funct: ast.FunctionDef) -> str:
        """
        Choose a parameter of the function if available, otherwise return a random integer as string.
        """
        params = self._funct_params_by_id.get(id(funct), [])
        if params:
            return random.choice(params)
        return str(random.randint(1, 100))
    
    
    # ---------- Main ----------
    def visit_Module(self, node: ast.Module):
        self._build_funct_params_by_id(node)
        self._insert_global_assignments(node, node)
        
        # Collect prepared garbage templates
        garbage_templates = get_garbage_templates()
        random.shuffle(garbage_templates)
        
        # Collect all function nodes in the module
        function_nodes = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
        if not function_nodes:
            return node
        
        # Randomly choose a function and insert the code snippet for each template
        for template in garbage_templates:
            funct = random.choice(function_nodes)
            param = self._choose_parameter_for_function(funct)
            snippet = template.format(parameter=param)
            self._insert_code_to_function_randomly(funct, snippet)
        
        return node
