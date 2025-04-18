from __future__ import annotations
from typing import List, Optional
import ast

op_map = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}


class Example:
    def __init__(self, func_name: str, input_structure: dict, operator: str = None, output: str = None,
                 tag: str = "correct", prefix: str = None, format: str = None) -> None:
        self.func_name = func_name
        self.input_structure = input_structure
        self.operator = operator
        self.output = output
        self.tag = tag
        self.prefix = prefix
        self.format = format

    def to_dict(self) -> dict:
        state = self.__dict__.copy()
        return state
    
    @classmethod
    def from_dict(cls, state: dict) -> "Example":
        return cls(**state)

    def get_function_call(self, hide_input: bool = False, follow_format: bool = True) -> str:
        """
        Convert the input_structure (nested dictionary of function calls) into a string representation.

        Args:
            hide_input (bool): If True, replaces arguments with "???".
            follow_format (bool): If True, processes the full nested structure.
                                  If False, only searches for `self.func_name`.
        """
        def reconstruct_call(structure: dict) -> str:
            if not structure:
                return ""
            for func_name, args in structure.items():
                # If follow_format is False, only return the target function
                if not follow_format and func_name != self.func_name:
                    for arg in args:
                        if isinstance(arg, dict):
                            return reconstruct_call(arg)
                    continue

                reconstructed_args = []
                for arg in args:
                    if isinstance(arg, dict):
                        reconstructed_args.append(reconstruct_call(arg))
                    else:
                        reconstructed_args.append("???" if hide_input else arg)

                return f"{func_name}({', '.join(reconstructed_args)})"

        return reconstruct_call(self.input_structure)

    def get_expression(self) -> str:
        """Reconstruct the full expression, including prefix."""
        prefix_part = f"{self.prefix} " if self.prefix else ""
        function_call = self.get_function_call()
        if self.operator and self.output is not None:
            return f"assert {prefix_part}{function_call} {self.operator} {self.output}"
        else:
            return f"assert {prefix_part}{function_call}"


def parse_example(target_func: str, expression: str, tag: str = "correct") -> Example:
    """
    Parse an expression and represent nested function calls as a dictionary.
    """

    def parse_function_calls(node: ast.AST) -> dict:
        """
        Recursively parse AST nodes to extract function calls and their arguments.
        """
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = f"{ast.unparse(node.func)}"
            else:
                func_name = "unknown"

            if func_name == target_func:
                # return {func_name: [ast.unparse(arg) for arg in node.args]}
                args = [ast.unparse(arg) for arg in node.args]
                kwargs = [f"{kw.arg} = {ast.unparse(kw.value)}" for kw in node.keywords]
                return {func_name: args + kwargs}

            args = []
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    args.append(parse_function_calls(arg))
                else:
                    args.append(ast.unparse(arg))
            kwargs = [
                f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
            return {func_name: args + kwargs}
        return {}

    expression = expression.replace("assert", "").strip()
    tree = ast.parse(expression, mode="eval").body

    operator = None
    output = None
    input_structure = {}
    prefix = None

    # Check prefix
    if isinstance(tree, ast.UnaryOp):
        if isinstance(tree.op, ast.Not):
            prefix = "not"
        elif isinstance(tree.op, ast.USub):
            prefix = "-"
        elif isinstance(tree.op, ast.UAdd):
            prefix = "+"
        tree = tree.operand

    # Check comparison and function call
    if isinstance(tree, ast.Compare):
        operator = op_map[type(tree.ops[0])]
        output = ast.unparse(tree.comparators[0])

        # Process the left side of the comparison (function call)
        if isinstance(tree.left, ast.Call):
            input_structure = parse_function_calls(tree.left)

    elif isinstance(tree, ast.Call):
        # If there's no comparison, process the function call directly
        input_structure = parse_function_calls(tree)
    
    format = "math_isclose" if "math.isclose" in expression else None

    return Example(
        func_name=target_func,
        input_structure=input_structure,
        operator=operator,
        output=output,
        tag=tag,
        prefix=prefix,
        format=format,
    )
