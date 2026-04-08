"""
Task 2: Simple Arithmetic Reasoning
Ask the model to solve a word problem and return a numeric answer.
"""

PROBLEM = (
    "A store sells apples for $0.50 each and oranges for $0.75 each. "
    "If a customer buys 4 apples and 3 oranges, how much do they spend in total? "
    "Reply with just the dollar amount, e.g. $3.25"
)

EXPECTED_ANSWER = "3.25"

def run(client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": PROBLEM}
        ],
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()
