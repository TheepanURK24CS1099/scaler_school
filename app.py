import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Inference API")

state = {"step": 0, "results": {}, "status": "idle"}


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
