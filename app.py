import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Inference API")

state = {"step": 0, "results": {}, "status": "idle"}


@app.get("/", response_class=HTMLResponse)
def home():
        return """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>OpenEnv Agent</title>
                <style>
                    body {
                        margin: 0;
                        min-height: 100vh;
                        display: grid;
                        place-items: center;
                        font-family: Inter, Arial, sans-serif;
                        background: linear-gradient(135deg, #0f172a, #1e293b);
                        color: #e2e8f0;
                    }
                    .card {
                        width: min(760px, calc(100vw - 32px));
                        padding: 36px;
                        border-radius: 24px;
                        background: rgba(15, 23, 42, 0.82);
                        border: 1px solid rgba(148, 163, 184, 0.2);
                        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
                    }
                    h1 {
                        margin: 0 0 12px;
                        font-size: clamp(2rem, 4vw, 3rem);
                    }
                    p {
                        margin: 0 0 18px;
                        font-size: 1.05rem;
                        line-height: 1.7;
                        color: #cbd5e1;
                    }
                    .endpoints {
                        display: grid;
                        gap: 10px;
                        margin-top: 18px;
                        padding: 0;
                        list-style: none;
                    }
                    .endpoint {
                        padding: 12px 16px;
                        border-radius: 14px;
                        background: rgba(30, 41, 59, 0.8);
                        border: 1px solid rgba(148, 163, 184, 0.16);
                        font-size: 1rem;
                    }
                    code {
                        color: #7dd3fc;
                        font-weight: 700;
                    }
                </style>
            </head>
            <body>
                <main class="card">
                    <h1>OpenEnv Agent is running</h1>
                    <p>This Space is live and serving the API endpoints used by the evaluator.</p>
                    <ul class="endpoints">
                        <li class="endpoint"><code>POST /reset</code></li>
                        <li class="endpoint"><code>POST /step</code></li>
                        <li class="endpoint"><code>GET /state</code></li>
                    </ul>
                </main>
            </body>
        </html>
        """


def get_client():
    if not API_BASE_URL or not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Missing API configuration. Set API_BASE_URL and HF_TOKEN (or OPENAI_API_KEY).",
        )
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


@app.post("/reset")
def reset():
    state["step"] = 0
    state["results"] = {}
    state["status"] = "idle"
    return {"status": "ok"}


@app.post("/step")
def step():
    from tasks import task1, task2, task3
    from graders import grader1, grader2, grader3

    task_map = [
        (task1, grader1, "task1"),
        (task2, grader2, "task2"),
        (task3, grader3, "task3"),
    ]

    current = state["step"]
    if current >= len(task_map):
        return {"status": "done", "results": state["results"]}

    task_mod, grader_mod, name = task_map[current]
    client = get_client()
    output = task_mod.run(client, MODEL_NAME)
    reward = grader_mod.grade(output)
    state["results"][name] = reward
    state["step"] += 1
    state["status"] = "running"

    return {"status": "ok", "task": name, "reward": reward}


@app.get("/state")
def get_state():
    return JSONResponse(content=state)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
