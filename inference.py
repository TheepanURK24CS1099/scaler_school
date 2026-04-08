import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_task(task_module, task_name):
    print(f"[STEP] running {task_name}")
    output = task_module.run(client, MODEL_NAME)
    return output

def grade_task(grader_module, output, task_name):
    print(f"[STEP] grading {task_name}")
    reward = grader_module.grade(output)
    assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range [0, 1]"
    print(f"[STEP] {task_name} reward: {reward:.4f}")
    return reward

def main():
    print("[START] inference")

    from tasks import task1, task2, task3
    from graders import grader1, grader2, grader3

    results = {}

    print("[STEP] loading model")

    # Task 1
    out1 = run_task(task1, "task1")
    r1 = grade_task(grader1, out1, "task1")
    results["task1"] = r1

    # Task 2
    out2 = run_task(task2, "task2")
    r2 = grade_task(grader2, out2, "task2")
    results["task2"] = r2

    # Task 3
    out3 = run_task(task3, "task3")
    r3 = grade_task(grader3, out3, "task3")
    results["task3"] = r3

    avg = sum(results.values()) / len(results)
    print(f"[STEP] average reward: {avg:.4f}")
    print(f"[STEP] results summary: {results}")

    print("[END] inference complete")
    return results

if __name__ == "__main__":
    main()
