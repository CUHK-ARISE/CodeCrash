GENERATE_INCORRECT_OUTPUT = """
Given the code snippet:
```{programming_language}
{code}
```
and the correct expression for the function call:
```{programming_language}
{expression}
```

Modify the output value to make it INCORRECT. The modification should introduce moderate changes, ensuring diversity and avoiding minimal adjustments. For example, if the output is a list, you can add new elements, remove elements, or modify the values of existing elements. However, the modification should still align logically with the code.
The purpose is to misleading people for getting correct answer.
Do NOT modify the function call and the input arguments!
Output the incorrect expression using the special tokens as follows: [EXPRESSION] assert <expression> [/EXPRESSION].

Example 1:
Given the function:
```{programming_language}
def f(n):
    return n
```
and the correct expression
```{programming_language}
assert f(17) == 17
```
Modify the expression such that it fails for the execution. 
You can modify the either the input arguments, or the output value, even both of them [EXPRESSION] assert f(10) == 20 [/EXPRESSION].


Example 2:
Given the function:
```{programming_language}
def f(s):
    return s + "a"
```
and the correct expression
```{programming_language}
assert f("x9j") == "x9ja"
```
Modify the expression such that it fails for the execution and output [EXPRESSION] assert f("x9j") == "x9j" [/EXPRESSION].
"""

