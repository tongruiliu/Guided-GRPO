import ast
from pathlib import Path


def test_sglang_backend_uses_dummy_actor_and_real_verifier_weights():
    tree = ast.parse(Path("verl/workers/rollout/vllm_rollout_spmd.py").read_text(encoding="utf-8"))
    engine_load_formats = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "SGLangEngineAdapter":
            continue
        for keyword in node.keywords:
            if keyword.arg == "load_format":
                engine_load_formats.append(ast.literal_eval(keyword.value))

    assert engine_load_formats == ["dummy", "auto"]


def test_sglang_backend_has_fsdp_sharding_manager_selection():
    worker_text = Path("verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    manager_text = Path("verl/workers/sharding_manager/fsdp_sglang.py").read_text(encoding="utf-8")

    assert "FSDPSGLangShardingManager" in worker_text
    assert 'rollout_backend == "sglang"' in worker_text
    assert "update_weights_from_tensor" in manager_text
