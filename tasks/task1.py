"""
Task 1: Text Summarization
Given a passage, ask the model to produce a concise summary.
"""

PASSAGE = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
    "as opposed to natural intelligence displayed by animals including humans. "
    "AI research has been defined as the field of study of intelligent agents, "
    "which refers to any system that perceives its environment and takes actions "
    "that maximize its chance of achieving its goals. The term 'artificial "
    "intelligence' had previously been used to describe machines that mimic and "
    "display human cognitive skills associated with the human mind, such as "
    "learning and problem-solving."
)

def run(client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following passage in one sentence:\n\n{PASSAGE}"
            }
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
