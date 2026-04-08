"""
Validate submission before uploading to Hugging Face Space.
Run: python validate_submission.py
"""

import os
import sys
import importlib
import subprocess
import yaml

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)


print("\n=== Submission Validator ===\n")

# 1. inference.py in root
print("[1] inference.py location")
check("inference.py exists in root", os.path.isfile("inference.py"))

# 2. Required files
print("\n[2] Required files")
for f in ["Dockerfile", "requirements.txt", "openenv.yaml", "app.py"]:
    check(f, os.path.isfile(f))

# 3. Tasks directory
print("\n[3] Tasks (minimum 3)")
for t in ["tasks/task1.py", "tasks/task2.py", "tasks/task3.py"]:
    check(t, os.path.isfile(t))

# 4. Graders directory
print("\n[4] Graders (minimum 3)")
for g in ["graders/grader1.py", "graders/grader2.py", "graders/grader3.py"]:
    check(g, os.path.isfile(g))

# 5. Grader reward range
print("\n[5] Grader reward range [0.0, 1.0]")
sys.path.insert(0, ".")
for name, mod_path in [("grader1", "graders.grader1"), ("grader2", "graders.grader2"), ("grader3", "graders.grader3")]:
    try:
        mod = importlib.import_module(mod_path)
        # test with a dummy output
        reward = mod.grade("test output") if name != "grader3" else mod.grade(["positive", "negative", "neutral"])
        in_range = isinstance(reward, float) and 0.0 <= reward <= 1.0
        check(f"{name} returns float in [0,1]", in_range, f"got {reward}")
    except Exception as e:
        check(f"{name} importable and callable", False, str(e))

# 6. openenv.yaml validity
print("\n[6] openenv.yaml structure")
try:
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    for key in ["name", "description", "tasks", "model", "endpoints"]:
        check(f"openenv.yaml has '{key}'", key in spec)
    check("endpoints has /reset", "reset" in spec.get("endpoints", {}))
    check("endpoints has /step", "step" in spec.get("endpoints", {}))
    check("endpoints has /state", "state" in spec.get("endpoints", {}))
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# 7. Dockerfile
print("\n[7] Dockerfile")
check("Dockerfile exists", os.path.isfile("Dockerfile"))
with open("Dockerfile") as f:
    df = f.read()
check("Dockerfile has CMD", "CMD" in df)
check("Dockerfile has EXPOSE or port", "EXPOSE" in df or "PORT" in df)

# 8. Structured log format in inference.py
print("\n[8] Structured log format in inference.py")
with open("inference.py") as f:
    inf = f.read()
check("inference.py prints [START]", "[START]" in inf)
check("inference.py prints [STEP]", "[STEP]" in inf)
check("inference.py prints [END]", "[END]" in inf)

# 9. OpenAI client usage
print("\n[9] OpenAI client usage")
check("inference.py uses OpenAI client", "from openai import OpenAI" in inf or "OpenAI(" in inf)

# 10. ENV variable loading
print("\n[10] Environment variable loading")
check("inference.py loads API_BASE_URL", "API_BASE_URL" in inf)
check("inference.py loads MODEL_NAME", "MODEL_NAME" in inf)
check("inference.py loads HF_TOKEN", "HF_TOKEN" in inf)

# 11. app.py endpoints
print("\n[11] app.py endpoint definitions")
with open("app.py") as f:
    appcontent = f.read()
check("app.py has /reset", '"/reset"' in appcontent or "'/reset'" in appcontent)
check("app.py has /step", '"/step"' in appcontent or "'/step'" in appcontent)
check("app.py has /state", '"/state"' in appcontent or "'/state'" in appcontent)

# Summary
print("\n" + "="*32)
if errors:
    print(f"\033[91m{len(errors)} check(s) FAILED:\033[0m")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\033[92mAll checks PASSED. Ready to submit!\033[0m")
