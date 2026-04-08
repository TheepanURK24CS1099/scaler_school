"""
Grader 2: Evaluates arithmetic reasoning.
Checks whether the model's answer contains the correct dollar amount ($3.25).
"""

import re

CORRECT_AMOUNT = 3.25

def grade(output: str) -> float:
    if not output or not isinstance(output, str):
        return 0.0

    # Extract all numbers from output
    numbers = re.findall(r"\d+\.?\d*", output)

    for num_str in numbers:
        try:
            val = float(num_str)
            if abs(val - CORRECT_AMOUNT) < 0.01:
                return 1.0
        except ValueError:
            continue

    return 0.0
