"""
Grader 1: Evaluates summarization quality.
Checks that the output is a non-empty single sentence containing key concepts.
"""

KEY_CONCEPTS = ["artificial intelligence", "machine", "intelligent", "ai", "learning"]

def grade(output: str) -> float:
    if not output or not isinstance(output, str):
        return 0.0

    text = output.lower().strip()

    # Must be non-empty
    if len(text) < 10:
        return 0.0

    score = 0.0

    # Reward for being a concise single sentence (no more than 60 words)
    words = text.split()
    if len(words) <= 60:
        score += 0.4

    # Reward for covering key concepts (up to 0.6)
    hits = sum(1 for kw in KEY_CONCEPTS if kw in text)
    concept_score = min(hits / 3, 1.0) * 0.6
    score += concept_score

    return round(min(score, 1.0), 4)
