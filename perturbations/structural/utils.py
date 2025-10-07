import ast

def parse_code(code: str) -> ast.AST:
    return ast.parse(code)


def unparse_code(tree: ast.AST) -> str:
    return ast.unparse(tree)


RESERVED_NAMES = {"self", "cls"}
def is_reserved(name: str) -> bool:
    return name in RESERVED_NAMES