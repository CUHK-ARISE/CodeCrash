from enum import Enum
from typing import Optional, Type, Union

class Perturbation(str, Enum):
    VAN = "VAN"         # Vanilla (no perturbation)
    REN = "REN"         # Renaming Entities
    RTF = "RTF"         # Reformatting Conditional Expressions
    GBC = "GBC"         # Inserting Garbage Code
    PSC_ALL = "ALL"     # Aggregated Structural Perturbation
    MCC = "MCC"         # Misleading Code Comments
    MPS = "MPS"         # Misleading Print Statements
    MHC = "MHC"         # Misleading Hint Comments

class Task(str, Enum):
    INPUT_PREDICTION = "input"     # Input prediction task
    OUTPUT_PREDICTION = "output"   # Output prediction task

class Mode(str, Enum):
    DIRECT = "direct"   # Direct inference
    COT = "cot"         # Chain of Thought prompting

def get_enum(value: str) -> Optional[Union[Perturbation, Task, Mode]]:
    if not isinstance(value, str):
        return None
    value_norm = value.strip().upper()
    for enum_cls in (Perturbation, Task, Mode):
        for member in enum_cls:
            if value_norm == member.name.upper() or value_norm == member.value.upper():
                return member
    return None