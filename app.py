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
                        font-family: Inter, Arial, sans-serif;
                        margin: 0;
                        min-height: 100vh;
                        display: grid;
                        place-items: center;
                        background: linear-gradient(135deg, #0f172a, #1e293b);
                        color: #e2e8f0;
                    }
                    .card {
                        max-width: 720px;
                        padding: 32px;
                        border-radius: 20px;
                        background: rgba(15, 23, 42, 0.75);
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
                        border: 1px solid rgba(148, 163, 184, 0.2);
                    }
                    h1 { margin-top: 0; font-size: 2rem; }
                    p { line-height: 1.6; color: #cbd5e1; }
                    code {
                        display: inline-block;
                        padding: 2px 8px;
                        border-radius: 999px;
                        background: rgba(51, 65, 85, 0.8);
                        color: #f8fafc;
                    }
                    ul { padding-left: 20px; }
                </style>
            </head>
            <body>
                <main class="card">
                    <h1>OpenEnv Agent is running</h1>
                    <p>This Space is live and serving the API endpoints used by the evaluator.</p>
                    <ul>
                        <li><code>POST /reset</code></li>
                        <li><code>POST /step</code></li>
                        <li><code>GET /state</code></li>
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
