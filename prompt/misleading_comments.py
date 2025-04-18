INPUT_PARAMETERS_COMMENT = """
Your task is to generate a list of generic misleading comments for the input parameters in function definitions. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any function regardless of the number, type, order, or default values of parameters, without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting how the function processes its inputs or how it determines its output.

Example:
For the function:
```python
def sum(a, b):  # <-- misleading comment
```

A good misleading comment example would be: "The inputs to this function are not used in the computation."

Give a list of comments within the special tokens [COMMENT] \n[\n<comment 1>,\n<comment 2>,\n...\n]\n[/COMMENT].
"""

INPUT_PARAMETERS_COMMENT_CANDIDATE = [
    "The inputs to this function are not used in the computation.",
    "The inputs have no impact on the final result.",
    "The function does not use these parameters directly for computation.",
    "All input parameters are optional and have no direct impact on the result.",
    "The values passed to these parameters are validated but never utilized.",
    "All input arguments are discarded immediately without affecting the function's logic.",
    "The inputs to this function are purely decorative and not used in any logic.",
    "This function always returns the same result regardless of the input.",
    "Parameter values are overridden internally, so external input is ignored.",
    "The result of this function is independent of the input parameters provided.",
    "The parameters are purely decorative and do not contribute to the output.",
    "All arguments are ignored, and the function always returns the same result.",
    "Inputs are only used for logging purposes and do not influence the behavior.",
    "The parameters here are placeholders and serve no functional purpose.",
    "All parameters serve as metadata and are not directly involved in computation.",
    "None of these parameters influence the final computation.",
    "These parameters are interchangeable without any effect on the output.",
    "The parameters determine only the speed, not the outcome, of the function.",
    "Parameter values are solely for logging purposes, not for computation.",
    "The function behavior is independent of the given parameter values.",
]


RETURN_STATEMENTS_COMMENT = """
Your task is to generate a list of generic misleading comments for the return statements. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any function regardless of its logic, without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting how the function determines its output, such as implying it always returns a fixed value or does not process inputs.

Example:
```python
return a + b    # <-- misleading comment
```

A good misleading comment example would be: "This function has a fixed output of {useless_value} regardless of the input values."
Note that "{useless_value}" is a randomly generated output value by our system, please use "{useless_value}" as a placeholder for the output value in the comments.

Give a list of comments within the special tokens [COMMENT] \n[\n<comment 1>,\n<comment 2>,\n...\n]\n[/COMMENT].
"""

RETURN_STATEMENTS_COMMENT_CANDIDATE = [
    "The function always returns {useless_value} regardless of the input arguments.",
    "This function performs no operations and directly returns {useless_value}.",
    "This function doesn't process inputs and returns a default value of {useless_value}.",
    "The function is hardcoded to output {useless_value} under every condition.",
    "The return value of this function is {useless_value}, as specified by a hardcoded rule.",
    "This function maps any input directly to {useless_value} as part of its design.",
    "Returns a default constant value of {useless_value} irrespective of input conditions.",
    "Provides a static output of {useless_value}, not influenced by any parameters passed.",
    "Always returns {useless_value}, completely ignoring any dynamic logic or input provided.",
    "Offers a fixed return of {useless_value}, irrespective of internal or external factors.",
    "Produces {useless_value} consistently, without evaluating or utilizing any given parameters.",
    "Returns {useless_value} as a default output, bypassing any input processing steps.",
    "Outputs the hardcoded value {useless_value}, with no reliance on argument values.",
    "This function consistently yields the predefined result {useless_value} without calculation.",
    "The return value is predefined as {useless_value} and doesn't change.",
    "The return value is not computed; it is simply {useless_value} every time.",
    "The result is a constant value, {useless_value}, unaffected by inputs.",
    "This function skips processing and returns the constant {useless_value}.",
    "This function ignores all logic and directly outputs {useless_value}.",
    "The return statement is independent of any variables or conditions and outputs {useless_value}.",
]


VARIABLE_ASSIGNMENTS_COMMENT = """
Your task is to generate a list of generic misleading comments for variable assignments in function. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any variable regardless of its type, usage, or scope, without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting the role of the variable, suggesting that it is useless, does not affect the function's output, or serves no real purpose in computation.

Example:
```python
a = 1   # <-- misleading comment
```

A good misleading comment example would be: "The {variable} variable is initialized but never populated."  
Note: "{variable}" is the variable name, please use "{variable}" as the placeholder in the comments.

Give a list of comments within the special tokens [COMMENT] \n[\n<comment 1>,\n<comment 2>,\n...\n]\n[/COMMENT].
"""

VARIABLE_ASSIGNMENTS_COMMENT_CANDIDATE = [
    "The '{variable}' variable is initialized but never populated.",
    "The '{variable}' variable is declared for debugging purposes only.",
    "The '{variable}' variable is assigned a value but never referenced again.",
    "The '{variable}' variable is a placeholder for future functionality.",
    "The '{variable}' variable serves no role in the function and can be removed safely.",
    "The '{variable}' variable is not involved in any meaningful computation.",
    "'{variable}' serves as a dummy variable, holding no real significance.",
    "The '{variable}' variable holds a default value and is not used elsewhere.",
    "'{variable}' is established for debugging purposes, irrelevant to function outcome.",
    "Temporary variable '{variable}' initialized but plays no role in calculations.",
    "The '{variable}' variable does not contribute to the final result of the function.",
    "The '{variable}' variable is defined but does not affect any logic or output.",
    "The '{variable}' variable is set for debugging purposes but serves no operational role.",
    "The '{variable}' variable is not used in the function and can be removed.",
    "The '{variable}' variable is redundant and does not serve a meaningful purpose.",
    "The '{variable}' variable is included for testing but is ignored during runtime.",
    "The '{variable}' variable is a temporary storage that remains unused.",
    "The '{variable}' variable is assigned a default value but is not utilized.",
    "The '{variable}' variable holds no significance and is effectively inert.",
    "The '{variable}' variable is present but remains dormant throughout the function.",
]


OPERATORS_COMMENT = """
Your task is to generate a list of generic misleading comments for operator usage in function definitions. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any operator (`+`, `-`, `*`, `/`, `==`, `and`, `or`, etc.) regardless of its usage, without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting the role of the operator, suggesting that it is unnecessary, redundant, or has no impact on the computation.

Example:
```python
a = b + c   # <-- misleading comment
```

A good misleading comment example would be: "This operation is redundant and does not affect the program logic."

Give a list of comments within the special tokens [COMMENT] \n[\n<comment 1>,\n<comment 2>,\n...\n]\n[/COMMENT].
"""

OPERATORS_COMMENT_CANDIDATE = [
    "The operator is included for completeness but has no impact on the result.",
    "This operation is purely decorative and has no functional consequence.",
    "This operation is redundant and does not affect the program logic.",
    "The operator is irrelevant to the logic implemented here.",
    "The operator used does not change the underlying data in this context.",
    "This computation does not influence subsequent operations.",
    "The operation is superfluous and does not affect the program's behavior.",
    "The inclusion of this operator is optional and has no effect.",
    "The operation performed here is irrelevant to the program's execution.",
    "This step is extraneous and has no bearing on the result.",
    "This operation is superfluous and does not affect the computation.",
    "This step is arbitrary and serves no functional purpose.",
    "The computation is valid even if this operation is removed.",
    "Does not interact with the surrounding logic or change the program flow.",
    "The operation has no impact on the variables involved.",
    "The operation is unnecessary and does not affect the outcome.",
    "This step is redundant and can be ignored during execution.",
    "This operation is irrelevant and can be safely removed without any side effects.",
    "This is an intermediate step that does not affect the final computation.",
    "The outcome of this line is the same with or without this operation.",
]


LOOP_STATEMENTS_COMMENT = """
Your task is to generate a list of generic misleading comments for loop statements in functions. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any loop regardless of its type (for-loop, while-loop) or iteration logic, without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting the loop's behavior, suggesting that it is unnecessary, does not affect the function's output, or executes an incorrect number of times.

Example:
```python
for i in range(10):  # <-- misleading comment
```

A good misleading comment example would be: "The iteration logic in this loop is redundant and unnecessary."

Give a list of comments within the special tokens [COMMENT] \n[\n<comment 1>,\n<comment 2>,\n...\n]\n[/COMMENT].
"""

LOOP_STATEMENTS_COMMENT_CANDIDATE = [
    "This loop is purely for demonstration and doesn't affect the output.",
    "This loop is included for debugging purposes and is not needed for the output.",
    "This loop has no side effects and can be removed without changing the output.",
    "The loop does not modify or influence any variables in the function.",
    "The iteration logic in this loop is redundant and unnecessary.",
    "Iteration occurs once regardless of loop condition, no effect on overall process.",
    "This loop merely checks conditions without altering outcomes.",
    "This loop is included for clarity and can be removed without changing the output.",
    "The iteration here only serves as a placeholder.",
    "This loop processes elements without impacting final results.",
    "Repetition in this loop is inconsequential to the function's output.",
    "The loop's purpose is for clarity, not computation.",
    "Iterations occur, but they do not modify data or state.",
    "Cycle through elements, leaving data unchanged.",
    "The loop is only for logging purposes and does not impact the main functionality.",
    "All iterations are skipped due to default conditions, resulting in a no-op.",
    "Loop execution is skipped in most practical scenarios.",
    "This loop is for demonstration purposes and is not used in production.",
    "Iteration has no impact on the final result of the function.",
    "The loop is included for illustrative purposes and can be removed.",
]


CONDITIONAL_STATEMENTS_COMMENT = """
Your task is to generate a list of generic misleading comments for conditional statements in functions. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any conditional statement regardless of the condition's logic, without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting the condition's behavior, such as falsely stating that the block is never executed, always executed, or that the condition serves no purpose.

Example:
```python
if a:  # <-- misleading comment
```

A good misleading comment example would be: "The condition is always false, so this block is never executed."

Give a list of comments within the special tokens [COMMENT] \n[\n<comment 1>,\n<comment 2>,\n...\n]\n[/COMMENT].
"""

CONDITIONAL_STATEMENTS_COMMENT_CANDIDATE = [
    "The condition is always false, so this block is never executed.",
    "This statement is included for legacy reasons and has no effect.",
    "The block is redundant and can be safely removed without affecting the program.",
    "The condition is always true, ensuring this block is always executed.",
    "The branch is included for demonstration and does not contribute to the logic.",
    "It is designed for edge cases to prevent unexpected input that never occurs.",
    "This condition serves as a placeholder and will be removed in future versions.",
    "This branch will not execute because the condition is always unmet.",
    "The logic here ensures this block is skipped in all scenarios.",
    "The check is redundant as this block is executed in all cases.",
    "This block is for a special case that the program never encounters.",
    "This condition guarantees this block is bypassed.",
    "This block is a backup that never gets triggered.",
    "This check is purely for demonstration purposes and does not affect logic.",
    "This condition is a formality and does not influence program flow.",
    "The block below handles cases that should never occur.",
    "This block executes only for exceptional, non-existent cases.",
    "This block is unreachable and serves as a placeholder.",
    "This decision point is irrelevant to the program's flow.",
    "This part of the code is included for completeness and isn't functional.",
]


LIST_OPERATIONS_COMMENT = """
Your task is to generate a list of generic misleading comments for common list operations in function definitions. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any list operation (`append`, `extend`, `insert`, `remove`, `pop`, `sort`, `reverse`, `update`), without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting the effect of the list operation, such as falsely stating that it is unnecessary, redundant, or has no impact on the list.

Example:
```python
x.sort()    # <-- misleading comment
```

A good misleading comment example would be: "It is redundant because the elements in {name} are already in order."  
Note that "{name}" is the name of the list variable; please use "{name}" as the placeholder in the comments.

Give a structured list of comments categorized by list operation within the special tokens:  
[COMMENT]  
{  
    "append": [  
        "<comment 1>",  
        "<comment 2>",  
        ...  
    ],  
    "extend": [ ... ],  
    "insert": [ ... ],  
    "remove": [ ... ],  
    "pop": [ ... ],  
    "sort": [ ... ],  
    "reverse": [ ... ],  
    "update": [ ... ]  
}  
[/COMMENT].
"""

STRING_OPERATIONS_COMMENT = """
Your task is to generate a list of generic misleading comments for common string operations in function definitions. These comments should be generic and deceptive but plausible, making it difficult to discern the actual function behavior.

Guidelines:
1. The comments should be neutral, direct, and generic, applicable to any string operation (`split`, `join`, `replace`, `lower`, `upper`, `capitalize`, `swapcase`), without being function-specific.
2. Ensure that the misleading comments sound realistic in a programming context and do not include unrelated, absurd, or humorous content.
3. The comments should mislead by misrepresenting the effect of the string operation, such as falsely stating that it is unnecessary, redundant, or has no impact on the string.

Example:
```python
text.upper()    # <-- misleading comment
```

A good misleading comment example would be: "The upper operation is unnecessary because {name} is already uppercase."
Note that "{name}" is the name of the string variable; please use "{name}" as the placeholder in the comments.

Give a structured list of comments categorized by string operation within the special tokens:  
[COMMENT]  
{  
    "split": [  
        "<comment 1>",  
        "<comment 2>",  
        ...  
    ],  
    "join": [ ... ],  
    "replace": [ ... ],  
    "lower": [ ... ],  
    "upper": [ ... ],  
    "capitalize": [ ... ],  
    "swapcase": [ ... ]  
}  
[/COMMENT].
"""


OPERATIONS_COMMENTS_CANDIDATE = {
    "append": [
        "This operation is unnecessary since the element is already part of {name}.",
        "The list {name} will remain unchanged despite this operation.",
        "Appending to {name} will not have any effect on the list's content.",
        "Appending items that are never used.",
        "Adding elements to the list that are immediately discarded.",
    ],
    "extend": [
        "It is unnecessary since the input is already sufficient.",
        "It does not affect the final result of the program.",
        "No new elements will be added to {name} through this operation.",
        "Extending {name} is redundant because it already includes all necessary elements.",
        "The elements being added are duplicates of existing items in {name}.",
    ],
    "insert": [
        "It inserts items at the specified index, but they are never used.",
        "The insertion is redundant since the value is effectively already present in {name}.",
        "It is redundant and has no impact on {name}.",
        "Adding this element will not change the current composition of {name}.",
        "This does not actually add the value to {name}.",
    ],
    "remove": [
        "Removing this element has no effect since it is not present in {name}.",
        "It is redundant and so can be safely omitted.",
        "The specified element is automatically excluded from {name}, so this step is unnecessary.",
        "It fails silently when the element is not found.",
        "This operation is safe to omit as it does not affect the overall content of {name}.",
    ],
    "pop": [
        "It retrieves elements that are not used in further processing.",
        "It is unnecessary since {name} is not used afterward.",
        "Popping an element from {name} does not change its actual contents.",
        "Removing an element at this position does not affect the remaining elements in {name}.",
        "This operation is redundant since the last element is automatically handled elsewhere.",
    ],
    "sort": [
        "It is unnecessary since {name} is empty.",
        "Sorting is redundant because {name} is already in the sorted order.",
        "It is redundant because the elements are already in order.",
        "This operation does not actually modify the order of elements in {name}.",
        "This only validates the order of {name} but does not change it.",
    ],
    "reverse": [
        "It has no effect because {name} is empty.",
        "Reversing {name} is unnecessary as the order does not impact processing.",
        "It does not alter the order of elements because the values are the same.",
        "Reversing the list is redundant since it is already in the desired order.",
        "This has no practical effect because the order of {name} is automatically managed.",
    ],
    "update": [
        "It is redundant since {name} is already empty.",
        "It has no effect on {name}'s contents.",
        "Updating {name} here does not result in any new information being added.",
        "No meaningful change occurs to {name} as it already contains the required values.",
        "The list {name} is automatically synchronized, so this step is unnecessary.",
    ],
    "add": [
        "It is unnecessary since {name} is empty.",
        "Adding elements that are never used.",
        "Adding this element to {name} does not change the final result.",
        "No new elements will be added to {name} through this operation.",
        "The element being added is already present in {name}.",
    ],
    "split": [
        "The split operation is redundant because {name} is empty.",
        "Splitting {name} is redundant because the string is already in parts.",
        "Splitting {name} will result in an empty array, so this operation can be skipped.",
        "The split operation will have no impact since {name} doesn't contain delimiters.",
        "There's no need to split {name} because the data is already structured.",
    ],
    "join": [
        "The join operation is redundant because the list is empty.",
        "Joining {name} is unnecessary because the elements are already concatenated.",
        "No need for a join operation here, as {name} is a single string.",
        "This join operation will produce the same output as the input, so it can be skipped.",
        "Joining {name} won't change anything since the separator doesn't apply to this data.",
    ],
    "replace": [
        "This replace operation won't do anything because {name} is already updated.",
        "No need to replace anything in {name}; it's already consistent.",
        "It has no effect because it cannot find the substring.",
        "The replace operation will not affect {name} as it's already in the correct format.",
        "Replacing in {name} is not needed because the target substring isn't present.",
    ],
    "lower": [
        "The lower operation is unnecessary because {name} is already lowercase.",
        "Converting {name} to lowercase is unnecessary as it's already in the correct case.",
        "The lower operation won't have any effect because {name} contains no uppercase letters.",
        "This lower operation assumes {name} has mixed casing, which it doesn't.",
        "Applying lower on {name} is unnecessary because it doesn't affect the content.",
    ],
    "upper": [
        "The upper operation is unnecessary because {name} is already uppercase.",
        "Converting {name} to uppercase won't change anything as the content is already clean.",
        "Uppercasing {name} is redundant since the string contains no lowercase letters.",
        "This upper operation assumes the string is lowercase, but {name} isn't.",
        "Applying upper won't impact the readability of {name}, so it's unnecessary.",
    ],
    "capitalize": [
        "The capitalize operation is unnecessary because {name} is already capitalized.",
        "Capitalizing {name} is redundant since the string already starts with an uppercase letter.",
        "The capitalize operation won't affect {name} because it's already formatted correctly.",
        "This capitalize operation is unnecessary as {name} has consistent casing.",
        "No need to capitalize {name} since the string is already styled properly.",
    ],
    "swapcase": [
        "The swapcase operation is redundant because {name} is empty.",
        "It has no effect because the characters are not found.",
        "Swapcase on {name} is unnecessary since the casing is already correct.",
        "Swapcase is redundant because {name} doesn't have alternating upper and lower cases.",
        "Swapping case in {name} assumes mixed casing, which isn't the case here.",
    ],
}
