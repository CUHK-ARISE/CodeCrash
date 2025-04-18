

# ======================================================== #

# Rephrase
REPHRASE_PROMPT = '''
Please rephrase the following content and output the rephrased content using the special tokens [START/rephrased] and [END/rephrased]:
"""
{content}
"""
'''

TRANSLATE_PROMPT = '''
Please translate the following content to {language} and output the translated content using the special tokens [START/translated] and [END/translated]:
"""
{content}
"""
'''


POSITIVE_COMMENT = '''
Please provide the comments for the following code snippet:
```{language}
{code}
```

The comments should be positive and should highlight the good practices followed in the code.

'''


GENERATE_INCORRECT_INPUT = """
Given the code snippet:
```{programming_language}
{code}
```
and the correct expression for the function call:
```{programming_language}
{expression}
```

Modify the input argument(s) to make it INCORRECT. The purpose is to mislead people to get a correct answer. Do NOT modify the output value!
Please think about how to modify the input arguments to make the expression incorrect step by step before arriving at an answer within the tokens [THOUGHT] and [/THOUGHT]
Output the incorrect expression using the special tokens as follows: [EXPRESSION] assert <expression> [/EXPRESSION].
Remember, the modification should introduce moderate changes, ensuring diversity and avoiding minimal adjustments.
However, if the function always returns the same value regardless of input, force an incorrect expression by modifying the arguments in a way that ensures failure (e.g., change an input's type, swap their order, or add/remove an argument).

Example 1:
Given the function:
```{programming_language}
def f(n):
    return n
```
and the correct expression:
```{programming_language}
assert f(17) == 17
```
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 
Since the function f(n) returns n, for f(??) to be equal to 17, the value of ?? should be 17.
Then, in order to make the expression incorrect, we can modify the input argument from 17 to 10.
[/THOUGHT]
[EXPRESSION] assert f(10) == 17 [/EXPRESSION].
"""


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
