import ast
from pathlib import Path


def test_local_verifier_loads_real_weights():
    rollout_path = Path("verl/workers/rollout/vllm_rollout_spmd.py")
    tree = ast.parse(rollout_path.read_text(encoding="utf-8"))

    load_formats = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "LLM":
            continue
        for keyword in node.keywords:
            if keyword.arg == "load_format":
                load_formats.append(ast.literal_eval(keyword.value))

    assert load_formats == ["dummy", "auto"]
