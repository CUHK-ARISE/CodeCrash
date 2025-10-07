from utils.format import Task

GENERATE_INCORRECT_INPUT = """
Given the code snippet:
```{programming_language}
{code}
```
and the correct expression for the function call:
```{programming_language}
{expr}
```

Modify the input argument(s) to make it INCORRECT. The purpose is to mislead people to get a correct answer. Do NOT modify the output value!
Please think about how to modify the input arguments to make the expression incorrect step by step before arriving at an answer within the tokens [THOUGHT] and [/THOUGHT]
Output the incorrect input using the special tokens as follows: [EXPR] incorrect_input = <incorrect input> [/EXPR].
Remember, the modification should introduce moderate changes, ensuring diversity and avoiding minimal adjustments.
However, if the function always returns the same value regardless of input, force an incorrect input by modifying the arguments in a way that ensures failure (e.g., change an input's type, swap their order, or add/remove an argument).

For Example:
Given the function:
```{programming_language}
def f(a, b):
    return a + b
```
and the correct expression:
```{programming_language}
assert f(17, 3) == 20
```

Example Output:
[THOUGHT]
Given that the correct expression is `assert f(17, 3) == 20`, we need to modify the input argument to make the expression incorrect.
Since the function f(a, b) returns a + b, for f(??, ??) to be equal to ??.
In order to make the expression incorrect, we can modify the input argument from 17 to 10 and 3 to 5.
Therefore, `f(10, 5) == 20` will be incorrect, the incorrect input should be 10 and 5.
[/THOUGHT]
[EXPR] incorrect_input = 10, 5 [/EXPR] (do not include the parentheses)
"""


GENERATE_INCORRECT_OUTPUT = """
Given the code snippet:
```{programming_language}
def f(a, b):
    return a + b
```
and the correct expression for the function call:
```{programming_language}
assert f(17, 3) == 20
```

Modify the output value to make it INCORRECT. The modification should introduce moderate changes, ensuring diversity and avoiding minimal adjustments. For example, if the output is a list, you can add new elements, remove elements, or modify the values of existing elements. However, the modification should still align logically with the code.
The purpose is to misleading people for getting correct answer.
```{programming_language}
{code}
```
and the correct expression for the function call:
```{programming_language}
{expr}
```

Modify the output value to make it INCORRECT. The modification should introduce moderate changes, ensuring diversity and avoiding minimal adjustments. For example, if the output is a list, you can add new elements, remove elements, or modify the values of existing elements. However, the modification should still align logically with the code.
The purpose is to misleading people for getting correct answer.
Please think about how to modify the input arguments to make the expression incorrect step by step before arriving at an answer within the tokens [THOUGHT] and [/THOUGHT]
Do NOT modify the function call and the input arguments!
Output the incorrect output using the special tokens as follows: [EXPR] incorrect_output = <incorrect output> [/EXPR].

For Example:
Given the function:
```{programming_language}
def f(n):
    return n
```
and the correct expression
```{programming_language}
assert f(17) == 17
```

Example Output:
[THOUGHT]
Given that the correct expression is `assert f(17) == 17`, we need to modify the output value to make the expression incorrect.
Since the function f(n) returns n, for f(??) to be equal to ??.
In order to make the expression incorrect, we can modify the output value from 17 to 20, which share the same type and are moderately different.
Therefore, `f(17) == 20` will be incorrect, the incorrect output should be 20.
[/THOUGHT]
[EXPR] incorrect_output = 20 [/EXPR]
"""


def get_incorrect_gen_prompt(code: str, expr: str, task: Task) -> str:
    if task == Task.INPUT_PREDICTION:
        return GENERATE_INCORRECT_INPUT.format(programming_language="python", code=code, expr=expr)
    elif task == Task.OUTPUT_PREDICTION:
        return GENERATE_INCORRECT_OUTPUT.format(programming_language="python", code=code, expr=expr)
    else:
        raise ValueError("task must be either 'Task.INPUT_PREDICTION' or 'Task.OUTPUT_PREDICTION'")