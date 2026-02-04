# Guided-GRPO

**Guided Reinforcement Learning for Multimodal Reasoning.**  
This repository contains Guided‑GRPO, a framework that injects **process‑level verification** into multimodal RL rollouts to stabilize training and reduce error propagation.

---

## Table of Contents
- [Key Ideas](#key-ideas)
- [Method Overview](#method-overview)
- [Code Map](#code-map)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Guided‑GRPO (Multi‑Turn Verifier)](#guided-grpo-multi-turn-verifier)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Reward Functions](#reward-functions)
- [Logging & Checkpoints](#logging--checkpoints)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Key Ideas
- **Guided Verifier**: a lightweight verifier monitors and corrects the policy during rollout, turning open‑loop exploration into guided navigation.
- **Process‑level feedback**: step‑wise guidance reduces cascading errors from early reasoning mistakes.
- **Guided‑GRPO**: integrates verifier feedback into GRPO without a separate value network.
- **Flexible verifier backends**: use a local verifier model or an OpenAI‑style HTTP service.

---

## Method Overview
Guided‑GRPO is an RL paradigm that turns open‑loop rollouts into **closed‑loop** reasoning:
1. **Verifier‑guided rollout**: a lightweight verifier interacts with the policy at each step, correcting errors and providing minimal guidance.
2. **Process‑level signals**: step‑wise feedback turns sparse outcome rewards into denser training signals.
3. **GRPO update**: group‑based advantage estimation updates the policy without a separate value network.

---

## Code Map
- `verl/workers/rollout/vllm_rollout_spmd.py`  
  Multi‑turn guided rollout (policy ↔ verifier) core implementation.
- `verl/workers/rollout/config.py`  
  Verifier configuration and default verifier prompt template.
- `verl/workers/reward/function.py`  
  Injects `verifier_hallucination_score` into reward inputs (optional).
- `verl/trainer/*`  
  Trainer, configs, and Ray orchestration.
- `examples/`  
  Ready‑to‑run configs and scripts.
- `examples/reward_function/*`  
  Reward function examples.

---

## Installation
**Python >= 3.9** is required.

```bash
pip install -r requirements.txt
```

(Optional) Docker environments are provided via `Dockerfile` and `Dockerfile.legacy`.

---

## Quick Start
Run standard GRPO training (no verifier):

```bash
python3 -m verl.trainer.main \
  config=examples/config.yaml
```

Example script: `examples/qwen2_5_vl_7b_geo3k_grpo.sh`

---

## Guided‑GRPO (Multi‑Turn Verifier)
Guided‑GRPO runs multi‑turn rollouts where the verifier interacts with the policy at each step.

### A) Local Verifier
Edit `examples/config_multi_turn.yaml`:
- `worker.rollout.verifier.enable: true`
- `worker.rollout.verifier.model_path: /path/to/verifier`
- optional `tokenizer_path`, `trust_remote_code`

Run:
```bash
python3 -m verl.trainer.main \
  config=examples/config_multi_turn.yaml
```

### B) HTTP Verifier (OpenAI‑style)
Edit `examples/config_multi_turn_http.yaml`:
- `worker.rollout.verifier.use_http: true`
- `worker.rollout.verifier.base_url: http(s)://...`
- `worker.rollout.verifier.model: <model_name>`
- `worker.rollout.verifier.api_key: <API_KEY>`

Run:
```bash
bash examples/debug_http_verifier_httpcfg_local8.sh
```

> The client sends requests to `POST {base_url}/chat/completions` with an OpenAI‑style payload.

---

## Configuration
Key fields (YAML):
- `data.max_prompt_length`, `data.max_response_length`
- `worker.rollout.n`, `worker.rollout.max_model_len`, `worker.rollout.max_num_batched_tokens`
- `worker.rollout.verifier.*` (enable, model_path / use_http, max_turns, prompt_template, etc.)
- `trainer.project_name`, `trainer.experiment_name`

**Important**: ensure
```
worker.rollout.max_num_batched_tokens >= data.max_prompt_length + data.max_response_length
```

---

## Data Format
Each sample should provide:
- `prompt_key` (default: `problem`)
- `answer_key` (default: `answer`)
- optional `images` / `videos`

For images, include `<image>` placeholders in the prompt text (one per image). The loader will attempt to auto‑insert if missing.

---

## Reward Functions
Reward functions live in `examples/reward_function/*.py`.

If `worker.rollout.verifier.enable_hallucination_score=true`, the verifier can append:
```
[SCORE] hallucination_detect=0|1
```
These are averaged into `verifier_hallucination_score` and passed to reward functions.

---

## Logging & Checkpoints
Logging backends (configurable): `file`, `tensorboard`, `wandb`, `console`.  
Checkpoints are saved to `trainer.save_checkpoint_path` (default: `checkpoints/<project>/<experiment>`).

---

## Limitations
- The verifier prompt template is task‑specific and should be tuned per domain.
- Multi‑turn rollouts add latency; plan GPU budgets accordingly.
- Verifier quality matters; weak verifiers can reduce training gains.

---

## Roadmap
- [ ] Additional benchmarks & scripts
- [ ] More verifier backends and deployment examples
- [ ] Improved logging and evaluation utilities

---

## Citation
If you use this code, please cite the paper draft:
```
@misc{guided_grpo,
  title={Guided-GRPO: Guided Reinforcement Learning for Multimodal Reasoning},
  note={Manuscript in preparation},
}
```

---

## License
Apache 2.0 (see `LICENSE`).
