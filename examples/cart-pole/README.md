# CartPole controller evolution

This example evolves a Gymnasium `CartPole-v1` controller while tuning its
numeric parameters with SciPy inside the evaluator.

## Install

From the repository root:

```bash
uv pip install -r examples/cart-pole/requirements.txt
```

## Check the initial tuned controller

```bash
PYTHONPATH=examples/cart-pole uv run python - <<'PY'
import json
from evaluator import evaluate

result = evaluate("examples/cart-pole/initial_program.py")
print(json.dumps(result.metrics, indent=2, sort_keys=True))
print(json.dumps(result.artifacts, indent=2, sort_keys=True))
PY
```

The evaluator tunes the candidate's declared parameter schema on five tuning
seeds, then scores it on ten separate validation seeds. The initial PID
controller should reach the 500-step limit on the nominal validation suite.

## Run OpenEvolve

```bash
openevolve-run \
  examples/cart-pole/initial_program.py \
  examples/cart-pole/evaluator.py \
  --config examples/cart-pole/config.yaml
```

The run writes checkpoints under the configured output directory. Each
checkpoint stores the best program and its metrics, including:

```text
combined_score
mean_survival
worst_case_survival
cvar_survival
success_rate
action_switch_rate
parameter_count
complexity_penalty
```

## Plot metric improvement

```bash
PYTHONPATH=examples/cart-pole uv run python \
  examples/cart-pole/visualize.py progress \
  --output-dir examples/cart-pole/openevolve_output \
  --destination examples/cart-pole/evolution_progress.png
```

The plot shows combined score, survival metrics, complexity penalty, and
action-switch rate over checkpoint iterations.

## Render the final controller

```bash
PYTHONPATH=examples/cart-pole uv run python \
  examples/cart-pole/visualize.py rollout \
  --program examples/cart-pole/openevolve_output/best/best_program.py \
  --destination examples/cart-pole/final_rollout \
  --seed 100
```

This produces:

```text
final_rollout/rollout.gif
final_rollout/rollout_trajectory.png
final_rollout/controller_params.json
```

The default rendered scenario is the nominal Suite 1 scenario. Use
`--low` and `--high` to visualize a wider initial-state scenario.
