"""
Grader 3: Evaluates sentiment classification accuracy.
Compares model labels to ground truth for each review.
"""

from tasks.task3 import REVIEWS

def grade(outputs: list) -> float:
    if not outputs or not isinstance(outputs, list):
        return 0.0

    ground_truth = [label for _, label in REVIEWS]

    if len(outputs) != len(ground_truth):
        return 0.0

    correct = 0
    for pred, truth in zip(outputs, ground_truth):
        # Accept if the truth label appears anywhere in the prediction
        if truth in pred.lower():
            correct += 1

    reward = correct / len(ground_truth)
    return round(reward, 4)
