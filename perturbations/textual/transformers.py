import ast
import random
from typing import Dict, List, Optional, Union

from perturbations.textual.utils import get_useless_value

from .misleading_messages import (
    INPUT_PARAMETERS_COMMENT_CANDIDATE,
    OPERATORS_COMMENT_CANDIDATE,
    OPERATIONS_COMMENTS_CANDIDATE,
    LOOP_STATEMENTS_COMMENT_CANDIDATE,
    RETURN_STATEMENTS_COMMENT_CANDIDATE,
    VARIABLE_ASSIGNMENTS_COMMENT_CANDIDATE,
    CONDITIONAL_STATEMENTS_COMMENT_CANDIDATE
)

CATALOG: Dict[str, List[str]] = {
    "input": INPUT_PARAMETERS_COMMENT_CANDIDATE,
    "return": RETURN_STATEMENTS_COMMENT_CANDIDATE,
    "variable": VARIABLE_ASSIGNMENTS_COMMENT_CANDIDATE,
    "operator": OPERATORS_COMMENT_CANDIDATE,
    "loop": LOOP_STATEMENTS_COMMENT_CANDIDATE,
    "conditional": CONDITIONAL_STATEMENTS_COMMENT_CANDIDATE,
    "operations": OPERATIONS_COMMENTS_CANDIDATE
}


class MisleadingInsertor(ast.NodeVisitor):
    """
    Insert Misleading Comments or Prints into Python code.
    
    Available:
        code: the Python source code as a string.
        mode: 'comment' (inline comment) or 'print' (print statement).
        output: expected output (literal string); affects return-comments.
        once: if True, each category inserts at most once.
        p: probability (0..1) to insert for each candidate site.
    """

    def __init__(
        self,
        code: str,
        mode: str = "comment",          # 'comment' | 'print'
        output: Optional[str] = None,   # literal string for expected output; affects return-comments
        once: bool = False,             # each category at most once
        p: float = 1.0,                 # probability to insert at each candidate site (controls density)
    ):
        self.code = code
        self.lines = code.split("\n")
        self.mode = mode
        self.once = once
        self.p = float(p)
        self.offset = 0
        self.parent = {}

        # For return message, we try to use the actual output value if provided; otherwise, just need a "useless value"
        self.output_value = ast.literal_eval(output) if output is not None else None
        self.output_type = type(self.output_value) if output is not None else None

        # Category guard (for once=True)
        self.contains = {
            "input": False,
            "return": False,
            "variable": False,
            "loop": False,
            "conditional": False,
            "operator": False,
            "operation": False
        }

    # ------------- Helpers -------------
    def _should_insert(self, category: str) -> bool:
        """ Decide whether to insert a comment/print for this category at this site. """
        if self.once and self.contains.get(category, False):
            return False
        return random.random() <= self.p
    
    def _pick(self, key: str) -> Optional[str]:
        """ Pick a random message from the catalog for this category. """
        arr = CATALOG[key]
        return random.choice(arr) if arr else None
    
    def _append_inline_comment(self, lineno: int, text: str):
        """ Insert for comment mode: append an inline comment to the specified line. """
        self.lines[lineno] = self.lines[lineno] + f"    # {text}"
    
    def _insert_print_line(self, lineno: int, col_offset: int, text: str):
        """ Insert for print mode: insert a print statement at the specified line with given indentation. """
        indent = " " * col_offset
        self.lines.insert(lineno, f'{indent}print("{text}")')
        self.offset += 1
    
    def _build_parent_map(self, node: ast.AST):
        for child in ast.iter_child_nodes(node):
            self.parent[child] = node
            self._build_parent_map(child)

    def _node_in_subtree(self, root: ast.AST, target: ast.AST) -> bool:
        return any(n is target for n in ast.walk(root))

    def _enclosing_header_if_in_test(self, node: ast.AST):
        cur = node
        while cur in self.parent:
            par = self.parent[cur]
            if isinstance(par, ast.If) and self._node_in_subtree(par.test, node):
                return par
            if isinstance(par, ast.While) and self._node_in_subtree(par.test, node):
                return par
            cur = par
        return None
    
    
    # ------------- Visitor -------------
    
    ## ------------- Function Definition -------------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Insert comments/prints at the start of a function definition to indicate input parameters.
        """
        if self._should_insert("input"):
            text = self._pick("input")
            if text:
                if self.mode == "comment":
                    # put a comment at the end of the function def line
                    self._append_inline_comment(node.lineno - 1 + self.offset, text)
                elif self.mode == "print":
                    # put a print at the first line inside function body
                    self._insert_print_line(node.lineno + self.offset, node.col_offset + 4, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["input"] = True
        self.generic_visit(node)
    
    
    ## ------------- Return Statement -------------
    def visit_Return(self, node: ast.Return):
        """
        Insert comments/prints before return statements to indicate the return value.
        """
        if self._should_insert("return"):
            _, useless = get_useless_value(self.output_type) if self.output_type else (None, "a default value")
            text_tmpl = self._pick("return")
            if text_tmpl:
                text = text_tmpl.format(useless_value=useless)
                if self.mode == "comment":
                    # put a comment at the end of the return line
                    self._append_inline_comment(node.lineno - 1 + self.offset, text)
                elif self.mode == "print":
                    # put print JUST ABOVE the return line
                    self._insert_print_line(node.lineno - 1 + self.offset, node.col_offset, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["return"] = True
        self.generic_visit(node)
    
    
    ## ------------- Variable Assignment & Operations -------------
    def visit_Assign(self, node: ast.Assign):
        if not node.targets:
            return self.generic_visit(node)
        # If the assignment is an operation (e.g., a = b + c)
        if isinstance(node.value, ast.BinOp) and self._should_insert("operator"):
            text = self._pick("operator")
            if text:
                if self.mode == "comment":
                    # put a comment at the end of the operation line
                    self._append_inline_comment(node.lineno - 1 + self.offset, text)
                elif self.mode == "print":
                    # put a print at the operation line
                    self._insert_print_line(node.lineno - 1 + self.offset, node.col_offset, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["operator"] = True
        
        # If the assignment is a simple variable assignment (e.g., x = 42)
        if isinstance(node.targets[0], ast.Name) and self._should_insert("variable"):
            var_name = node.targets[0].id
            text_tmpl = self._pick("variable")
            if text_tmpl:
                text = text_tmpl.format(variable=var_name)
                if self.mode == "comment":
                    # put a comment at the end of the assignment line
                    self._append_inline_comment(node.lineno - 1 + self.offset, text)
                elif self.mode == "print":
                    # put a print at the assignment line
                    self._insert_print_line(node.lineno - 1 + self.offset, node.col_offset, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["variable"] = True
        self.generic_visit(node)
    
    
    ## ------------- Augment Assignment & Operations -------------
    def visit_AugAssign(self, node: ast.AugAssign):
        """
        Insert comments/prints for augmented assignments (e.g., x += 1) to indicate the operation.
        """
        if self._should_insert("operator"):
            text = self._pick("operator")
            if text:
                if self.mode == "comment":
                    # put a comment at the end of the operation line
                    self._append_inline_comment(node.lineno - 1 + self.offset, text)
                elif self.mode == "print":
                    # put a print at the operation line
                    self._insert_print_line(node.lineno - 1 + self.offset, node.col_offset, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["operator"] = True
        self.generic_visit(node)
    
    
    ## ------------- Loop Constructs -------------
    def _visit_loop_like(self, node):
        """
        Insert comments/prints for loop constructs (e.g., for/while) to indicate the operation.
        """
        if self._should_insert("loop"):
            text = self._pick("loop")
            if text and node.body:
                first_body_line = node.body[0].lineno - 1 + self.offset
                indent = " " * node.col_offset
                if self.mode == "comment":
                    self.lines.insert(first_body_line, f"{indent}# {text}")
                    self.offset += 1
                elif self.mode == "print":
                    body_indent = node.body[0].col_offset if node.body else node.col_offset + 4
                    self._insert_print_line(first_body_line, body_indent, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["loop"] = True
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        self._visit_loop_like(node)
    
    def visit_While(self, node: ast.While):
        self._visit_loop_like(node)
    
    
    ## ------------- Conditional Constructs -------------
    def visit_If(self, node: ast.If):
        """
        Insert comments/prints for conditional constructs (e.g., if/elif/else) to indicate the operation.
        """
        if self._should_insert("conditional"):
            text = self._pick("conditional")
            if text:
                if self.mode == "comment":
                    self._append_inline_comment(node.lineno - 1 + self.offset, text)
                elif self.mode == "print":
                    # self._insert_print_line(node.lineno + self.offset, node.col_offset + 4, text)
                    if node.body:
                        first = node.body[0]
                        insert_at = first.lineno - 1 + self.offset
                        self._insert_print_line(insert_at, first.col_offset, text)
                    else:
                        self._insert_print_line(node.lineno + self.offset, node.col_offset + 4, text)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                self.contains["conditional"] = True
        self.generic_visit(node)
    
    
    ## ------------- General Function Call & Operations -------------
    def visit_Call(self, node: ast.Call):
        """
        Insert comments/prints for general function calls to indicate the operation.
            e.g., .append/.extend/.pop/.replace/.isdigit() ...
        """
        ops_catalog = CATALOG["operations"]
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ops_catalog and self._should_insert("operation"):
                cand = ops_catalog[method_name]
                if cand:
                    obj_name = node.func.value.id if isinstance(node.func.value, ast.Name) else "the object"
                    text = random.choice(cand).format(name=obj_name)

                    if self.mode == "comment":
                        # 註解放同一行末端，安全
                        self._append_inline_comment(node.lineno - 1 + self.offset, text)

                    elif self.mode == "print":
                        # ✅ 關鍵：若此 Call 在 if/elif/while 的 test 中，把 print 插到對應 body 的第一行
                        header = self._enclosing_header_if_in_test(node)
                        if header is not None:
                            body = header.body
                            if body:
                                first = body[0]
                                insert_line = first.lineno - 1 + self.offset
                                indent_cols = first.col_offset
                                self._insert_print_line(insert_line, indent_cols, text)
                            else:
                                # 理論上不會，但保底：放到 header 之後一行
                                self._insert_print_line(header.lineno + self.offset, header.col_offset + 4, text)
                        else:
                            # 原策略：取當前行的實際前導空白做縮排，印在這一行「上方」
                            line_text = self.lines[node.lineno - 1 + self.offset]
                            leading_spaces = len(line_text) - len(line_text.lstrip(' '))
                            self._insert_print_line(node.lineno - 1 + self.offset, leading_spaces, text)
                    else:
                        raise ValueError(f"Unknown mode: {self.mode}")

                    self.contains["operation"] = True

        self.generic_visit(node)
    
    
    # ------------- Runner -------------
    def run(self) -> str:
        tree = ast.parse(self.code)
        self._build_parent_map(tree)
        self.visit(tree)
        out = "\n".join(self.lines)
        if self.mode == "print":
            return ast.unparse(ast.parse(out)).strip()
        return out.strip()


class FunctionHintInserter:
    """
    Insert a raw string at specific locations for a target function.

    mode="input"  -> append to the function 'def' line as a trailing comment
    mode="output" -> if exactly one 'return', append to that line as a trailing comment;
                     otherwise insert a new commented line near the end of the function body.

    NOTE: The provided 'text' is inserted verbatim (no repr / eval), but always as a comment,
          so it won't break syntax even if it's like "42" or "f(a=1, b=2)".
    """

    def __init__(self, code: str, target_function: str, text: str):
        self.code = code
        self.target_function = target_function
        self.text = text
    
    
    # ---------- Helpers ----------
    def _find_target_func(self, tree: ast.AST) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
        """ Find the target function definition in the AST. """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == self.target_function:
                return node
        return None
    
    def _last_body_lineno_approx(self, funct: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """ Approximate the last line number inside the function. """
        all_nodes: List[ast.AST] = list(ast.walk(funct))
        last = max((getattr(n, "lineno", 0) for n in all_nodes), default=funct.lineno)
        return max(1, last)
    
    
    # ---------- Insertion Modes ----------
    def _run_input_mode(self, lines: List[str], funct: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """ Insert hint as an inline comment at the function definition. """
        idx = funct.lineno - 1
        lines[idx] = lines[idx].rstrip("\n") + f"    # {self.text}\n"
        return lines
    
    def _run_output_mode(self, lines: List[str], funct: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """ Insert hint as an inline comment at the return statement or near the end of the function body. """
        returns = [n for n in ast.walk(funct) if isinstance(n, ast.Return)]
        if len(returns) == 1:
            # If there's exactly one return statement, append to that line
            r = returns[0]
            ridx = r.lineno - 1
            lines[ridx] = lines[ridx].rstrip("\n") + f"    # {self.text}\n"
            return lines
        else:
            # Otherwise, insert a new line before the end of the function body
            body_indent = " " * (funct.col_offset + 4)
            last_line = self._last_body_lineno_approx(funct)
            insert_at = max(0, min(last_line, len(lines)))
            lines.insert(insert_at, f"{body_indent}# {self.text}\n")
            return lines
    
    
    # ---------- Runner ----------
    def run(self, mode: str) -> str:
        tree = ast.parse(self.code)
        funct = self._find_target_func(tree)
        
        if funct is None:
            raise ValueError(f"Function '{self.target_function}' not found in the provided code.")

        lines = self.code.splitlines(True)

        if mode == "input":
            lines = self._run_input_mode(lines, funct)
        elif mode == "output":
            lines = self._run_output_mode(lines, funct)
        else:
            raise ValueError("mode must be either 'input' or 'output'")

        return "".join(lines).strip()
