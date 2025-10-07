import ast
import random
from typing import List

from perturbations.structural.renaming import get_parameters
from perturbations.structural.utils import is_reserved


# ---------- Templates ----------
GEN_TEMPLATES = [
    "not not ({cond})",
    "({cond}) or False",
    "({cond}) and True",
    "(_ := ({cond},)[0])",
]

COMP_TEMPLATES = [
    "(_ := ({cond},)[0])",
    "(lambda: {cond})()",
    "({cond}) == ({para} == {para})",
    "({cond}) != ({para} != {para})",
    "bool(-~({cond})) == ({cond})",
    "bool(int({cond}))",
]

CONST_TEMPLATES = [
    "(_ := ({cond},)[0])",
    "(lambda: {cond})()",
    "({cond}) == ({para} == {para})",
    "({cond}) != ({para} != {para})",
    "bool(-~({cond})) == ({cond})",
    "eval(str({cond}))",
]


class ConditionReformatter(ast.NodeTransformer):
    """
    Reformat conditions in if/while statements using randomized templates.
    Keeps track of enclosing function parameters via get_parameters.
    """
    def __init__(self, code: str):
        super().__init__()
        self.code = code
        self.funct_stack: List[str] = []
    
    
    # ---------- Helpers ----------
    def _choose_templates(self, test: ast.AST) -> List[str]:
        if isinstance(test, ast.Compare):
            return COMP_TEMPLATES
        if isinstance(test, ast.Constant) and isinstance(test.value, bool):
            return CONST_TEMPLATES
        return GEN_TEMPLATES
    
    def _rewrite_test(self, node: ast.AST) -> ast.AST:
        cond_src = ast.unparse(node)
        templates = self._choose_templates(node)
        
        params = get_parameters(self.code, target_funct=self.funct_stack[-1]) if self.funct_stack else []
        params = [p for p in params if not is_reserved(p)]
        candidates = (params if params else []) + ["True", "False"]
        
        chosen_param = random.choice(candidates)
        template = random.choice(templates)
        new_src = template.format(cond=cond_src, para=chosen_param)
        
        new_expr = ast.parse(new_src, mode="eval")
        return new_expr.body
    
    
    # ---------------- Visitors ----------------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.funct_stack.append(node.name)
        node = self.generic_visit(node)
        self.funct_stack.pop()
        return node
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.funct_stack.append(node.name)
        node = self.generic_visit(node)
        self.funct_stack.pop()
        return node
    
    def visit_If(self, node: ast.If):
        node.test = self._rewrite_test(node.test)
        node.body = [self.visit(n) for n in node.body]
        node.orelse = [self.visit(n) for n in node.orelse]
        return node
    
    def visit_While(self, node: ast.While):
        node.test = self._rewrite_test(node.test)
        node.body = [self.visit(n) for n in node.body]
        node.orelse = [self.visit(n) for n in node.orelse]
        return node
