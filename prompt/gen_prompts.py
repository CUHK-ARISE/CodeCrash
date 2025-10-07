from utils.format import Task, Mode

RUN_PREDICTION_INPUT_DIRECT = """
Given the code snippet:
```{programming_language}
{code}
```
and output value:
```{programming_language}
{output}
```
Please predict the input arguments for the function `{function_name}` that result in the output value `{output}` and output your prediction using the special tokens [ANSWER] {function_name}(??) == {output} [/ANSWER].
There may be multiple answers, but you should only output one. Ensure the provided expression syntax is correct!
Do NOT output any extra information.

Example 1:
Given the code snippet:
```{programming_language}
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
```
and output value:
```{programming_language}
3
```
The input arguments for `f` that result in the output value of `3` are `["mq", "px", "zy"]`, then output your prediction:
[ANSWER] f(["mq", "px", "zy"]) == 3 [/ANSWER]

Example 2:
Given the code snippet:
```{programming_language}
def f(s1, s2):
    return s1 + s2
```
and output value:
```{programming_language}
"banana"
```
The input arguments for `f` that result in the output value of `"banana"` are `"ba", "nana"`, then output your prediction:
[ANSWER] f("ba", "nana") == "banana" [/ANSWER]
"""


RUN_PREDICTION_INPUT_COT = """
Given the code snippet:
```{programming_language}
{code}
```
and output value:
```{programming_language}
{output}
```
Please predict the input arguments for the function `{function_name}` that result in the output value `{output}` step by step before arriving at an answer within the tokens [THOUGHT] and [/THOUGHT], and output your prediction using the special tokens [ANSWER] {function_name}(??) == {output} [/ANSWER].
There may be multiple answers, but you should only output one. Ensure the provided expression syntax is correct!
Do NOT output any extra information.

For example:
Given the code snippet:
```{programming_language}
def f(x):
    return x + 1
```
and the output value:
```{programming_language}
17
```

[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]

Thus, the input arguments for the function `f` that result in the output value `17` is `16`, then output your prediction:
[ANSWER] f(16) == 17 [/ANSWER]
"""


RUN_PREDICTION_OUTPUT_DIRECT = """
Given the code snippet:
```{programming_language}
{code}
```
and the function call with input arguments:
```{programming_language}
{function_call}
```
Predict the exact output value for `{function_call}` and output your prediction using the special tokens [ANSWER] {function_call} == ?? [/ANSWER].
Ensure the provided expression syntax is correct!
Do NOT output any extra information.

Example 1:
Given the code snippet:
```{programming_language}
def f(n):
    return n
```
and the function call with input arguments:
```{programming_language}
f(17)
```
The output value for `f(17)` is 17, then output your prediction:
[ANSWER] f(17) == 17 [/ANSWER]

Example 2:
Given the code snippet:
```{programming_language}
def f(s):
    return s + "a"
```
and the function call with input arguments:
```{programming_language}
f("x9j")
```
The output value for `f("x9j")` is "x9ja", then output your prediction:
[ANSWER] f("x9j") == "x9ja" [/ANSWER]
"""


RUN_PREDICTION_OUTPUT_COT = """
Given the code snippet:
```{programming_language}
{code}
```
and the function call with input arguments:
```{programming_language}
{function_call}
```
Predict the exact output value for `{function_call}`, execute the program step by step before arriving at an answer within the tokens [THOUGHT] and [/THOUGHT], and output your prediction using the special tokens [ANSWER] {function_call} == ?? [/ANSWER].
Ensure the provided expression syntax is correct!
Do NOT output any extra information.

For example:
Given the code snippet:
```{programming_language}
def f(s):
    s = s + s
    return "b" + s + "a"
```
and the input arguments:
```{programming_language}
f("hi")
```

[THOUGHT]
Let's execute the code step by step:
1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]

Thus, the output value for `f("hi")` is "bhihia", then output your prediction:
[ANSWER] f("hi") == "bhihia" [/ANSWER]
"""


def get_run_prompt(dataset: str, task: Task, infer_mode: Mode, **kwargs) -> str:
    if dataset in ["lcb", "crux", "cruxeval", "LCB-Execution"]:
        if task == Task.INPUT_PREDICTION:
            try:
                code = kwargs["code"]
                function_name = kwargs["function_name"]
                output = kwargs["output"]
            except KeyError as e:
                raise ValueError(f"Missing required argument for input prompt: {e}")
            if infer_mode == "direct":
                return RUN_PREDICTION_INPUT_DIRECT.format(
                    programming_language="python",
                    code=code,
                    function_name=function_name,
                    output=output,
                )
            elif infer_mode == "cot":
                return RUN_PREDICTION_INPUT_COT.format(
                    programming_language="python",
                    code=code,
                    function_name=function_name,
                    output=output,
                )
            else:
                raise ValueError(f"Invalid infer_mode: {infer_mode}")

        elif task == Task.OUTPUT_PREDICTION:
            try:
                code = kwargs["code"]
                function_call = kwargs["function_call"]
            except KeyError as e:
                raise ValueError(f"Missing required argument for output prompt: {e}")
            if infer_mode == "direct":
                return RUN_PREDICTION_OUTPUT_DIRECT.format(
                    programming_language="python",
                    code=code,
                    function_call=function_call
                )
            elif infer_mode == "cot":
                return RUN_PREDICTION_OUTPUT_COT.format(
                    programming_language="python",
                    code=code,
                    function_call=function_call
                )
            else:
                raise ValueError(f"Invalid infer_mode: {infer_mode}")
        else:
            raise ValueError(f"Invalid Task: {task.value}")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")