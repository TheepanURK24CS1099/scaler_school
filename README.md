# LLM Evaluation Suite

A three-task LLM evaluation suite for Hugging Face Space deployment.

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   API_BASE_URL=https://api-inference.huggingface.co/v1
   MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

```bash
python inference.py
```

Expected output:
```
[START] inference
[STEP] loading model
[STEP] running task1
[STEP] grading task1
[STEP] task1 reward: 0.8000
...
[END] inference complete
```

## API Server

```bash
python app.py
```

Endpoints:
- `POST /reset` — reset state
- `POST /step` — run next task
- `GET /state` — current state

## Validation

```bash
python validate_submission.py
```

## Docker

```bash
docker build -t llm-eval .
docker run -p 7860:7860 --env-file .env llm-eval
```

## Tasks

| Task | Description | Grader |
|------|-------------|--------|
| task1 | Text summarization | Key concept coverage |
| task2 | Arithmetic reasoning | Exact numeric match |
| task3 | Sentiment classification | Accuracy across 3 reviews |

All rewards are in `[0.0, 1.0]`.
