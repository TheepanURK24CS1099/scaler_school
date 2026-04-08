"""
Task 3: Sentiment Classification
Ask the model to classify the sentiment of a review.
"""

REVIEWS = [
    ("This product is absolutely amazing! Best purchase I've ever made.", "positive"),
    ("Terrible quality. Broke after one day. Total waste of money.", "negative"),
    ("It's okay, nothing special but gets the job done.", "neutral"),
]

def run(client, model_name):
    outputs = []
    for review_text, _ in REVIEWS:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Classify the sentiment of this review as exactly one word: "
                        f"positive, negative, or neutral.\n\nReview: {review_text}"
                    )
                }
            ],
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().lower()
        outputs.append(label)
    return outputs
