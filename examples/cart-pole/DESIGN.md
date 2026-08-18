# CartPole + AlphaEvolve Design

This example will use OpenEvolve to evolve the *controller structure* for
Gymnasium's `CartPole-v1`. Numerical parameter tuning is performed by the
evaluator, not by the LLM.

## Answers to the design questions

### When are parameters tuned?

Parameters must be tuned during evaluation of every candidate, because the
tuned candidate score is what determines selection, archive placement, and
the next LLM prompt. If tuning happened only after the final candidate was
selected, evolution would optimize the wrong objective: it would compare
untuned or inconsistently tuned controllers.

The lifecycle is therefore:

```text
LLM proposes controller code
        |
        v
evaluator imports and validates candidate
        |
        v
numerical optimizer searches fixed parameter bounds
        |
        v
roll out the tuned controller on fixed scenarios
        |
        v
return combined_score + metrics + artifacts
        |
        v
OpenEvolve selects and evolves candidates
```

There should also be a final, more expensive retuning step after evolution:

1. Tune the initial PID baseline with the full tuning budget.
2. Tune each candidate during evolution with a small deterministic budget.
3. Retune the selected best controller with a larger budget and fresh seeds.
4. Report its held-out validation and test performance.

This preserves a fair evolutionary signal while producing a stronger final
result. The evolved program must not be allowed to change the global tuning
budget, scoring rules, or evaluation seeds. It may declare a controller-
specific parameter schema, subject to evaluator validation. This is necessary
because the best controller may not be a PID controller.

### Does Gymnasium provide visualization?

Yes. CartPole supports the normal Gymnasium rendering modes. Use:

```python
env = gym.make("CartPole-v1", render_mode="human")
```

for interactive display, or:

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
```

to collect frames for plots, GIFs, or videos. Gymnasium also provides
`RecordVideo` for episode videos and `RenderCollection` for collecting rendered
frames. These are appropriate for selected controllers, not for every
candidate evaluation. See the [Gymnasium rendering wrappers documentation](https://gymnasium.farama.org/api/wrappers/misc_wrappers/).

Gymnasium rendering gives us the CartPole animation. It does not provide the
analysis views we need, such as angle/velocity plots, action traces, score
distributions, or comparisons across evolved controllers. Those belong in a
local `visualize.py` script.

## Proposed files

```text
examples/cart-pole/
  DESIGN.md
  initial_program.py       # PID controller; only this code is evolved
  evaluator.py             # import, tune, roll out, score
  scenarios.py             # fixed scenario and seed definitions
  visualize.py             # plots, GIF/video generation, comparisons
  config.yaml              # OpenEvolve configuration
  requirements.txt         # gymnasium, scipy, matplotlib, imageio
```

The first implementation should remain an example and should not require
changes to OpenEvolve's core evolution loop.

## Candidate API

The candidate program should expose:

```python
def make_controller(params):
    return Controller(params)


def parameter_spec():
    return [
        {"name": "kp_theta", "low": -30.0, "high": 30.0},
        {"name": "kd_theta", "low": -10.0, "high": 10.0},
    ]


class Controller:
    def reset(self):
        pass

    def act(self, observation):
        # Return the discrete Gymnasium action: 0 or 1.
        pass
```

The initial controller is a PID-like law over the four observations:

```text
u = kp_x * cart_position
  + kd_x * cart_velocity
  + kp_theta * pole_angle
  + kd_theta * pole_angular_velocity

action = 1 if u >= threshold else 0
```

The LLM can evolve the structure: feature combinations, nonlinearities,
filters, state, gain scheduling, switching logic, and thresholds. The numeric
values are supplied by the evaluator as `params`.

The evaluator must not assume that every candidate has four PID parameters.
Examples of valid controller families include:

```text
PID:                 kp_x, kd_x, kp_theta, kd_theta
linear state space:  four feedback gains
nonlinear:           angle, angle^2, angle^3, and velocity gains
gain scheduled:      separate gains for small and large pole angles
filtered:            controller gains plus observation-filter parameters
parameter-free:      parameter_spec() returns an empty list
```

The LLM discovers the controller structure and parameterization; numerical
optimization selects values for the declared parameterization.

## Numerical tuning contract

The evaluator asks each candidate for its parameter schema:

```python
def parameter_spec():
    return [
        {"name": "kp_x", "low": -10.0, "high": 10.0},
        {"name": "kd_x", "low": -10.0, "high": 10.0},
        {"name": "kp_theta", "low": -30.0, "high": 30.0},
        {"name": "kd_theta", "low": -10.0, "high": 10.0},
        {"name": "threshold", "low": -2.0, "high": 2.0},
    ]
```

The evaluator converts the ordered schema into optimizer bounds and passes a
named dictionary to the candidate:

```python
spec = candidate.parameter_spec()
bounds = [(p["low"], p["high"]) for p in spec]
result = differential_evolution(objective, bounds=bounds, ...)
params = dict(zip((p["name"] for p in spec), result.x))
controller = candidate.make_controller(params)
```

The evaluator validates every schema:

- names are unique valid identifiers;
- values are finite numeric scalars;
- `low < high`;
- parameter count is below a configured maximum, initially perhaps 16;
- bounds are within global safety limits;
- the controller accepts the resulting parameter dictionary;
- actions are always valid `0` or `1`.

For example, the evaluator can enforce:

```python
MAX_PARAMETERS = 16
MAX_TUNING_EVALUATIONS = 250
GLOBAL_ABS_PARAMETER_LIMIT = 100.0
```

This allows new controller designs without allowing a candidate to make
tuning arbitrarily expensive. A parameter-free candidate returns `[]` and is
evaluated directly with `make_controller({})`.

The evaluator should reject candidates that return a different schema on
successive calls, declare hundreds of parameters, or use parameter values to
inspect tuning seeds. The schema is part of the evolved design, but the
optimizer, budget, scenario definitions, and scoring remain evaluator-owned.

The exact bounds should be calibrated against the PID baseline. The initial
tuner can use `scipy.optimize.differential_evolution`; a later local
refinement can use `scipy.optimize.minimize`.

The optimizer objective should be cheap and deterministic:

```text
objective(params) = negative mean survival on tuning seeds
```

Use the same tuning seeds for every candidate in a given evolution run. This
is common-random-number evaluation and reduces noise in selection.

The evaluator should record:

```text
tuned_params
tuning_objective
tuning_evaluations
tuning_time_seconds
```

These are diagnostics, not additional objectives.

## Evaluation scenarios

Use a staged suite. Begin with a small nominal experiment so that controller
and evaluator bugs are easy to diagnose, then increase difficulty only after
the baseline and evolution loop work reliably.

### Suite 1: Minimal nominal benchmark

This is the first implementation and the initial evolution objective:

- standard `CartPole-v1` physics;
- default reset range `[-0.05, 0.05]`;
- 5 fixed tuning seeds;
- 10 separate validation seeds;
- 500-step episode limit;
- no observation noise or action delay.

Tune parameters on the five tuning seeds and calculate the candidate's
`combined_score` on the ten validation seeds. The validation seeds must not be
used by the numerical optimizer.

### Suite 2: More nominal seeds

Once Suite 1 is stable:

- retain the same default reset range;
- increase to 20–50 tuning seeds for final retuning;
- use 50–100 held-out validation seeds;
- report mean, median, worst-case, CVaR, and success rate.

This primarily reduces stochastic evaluation noise without changing the
control problem.

### Suite 3: Wider initial conditions

After nominal performance is strong, add reset ranges:

```text
easy:   [-0.05, 0.05]
medium: [-0.10, 0.10]
hard:   [-0.15, 0.15]
```

Gymnasium's CartPole reset API supports changing the initial-state bounds via
`reset(options={"low": ..., "high": ...})`. Keep the medium and hard ranges
out of tuning initially; use them for validation and final testing.

### Suite 4: Recovery and sensor robustness

Add controlled tests for:

- initial states near the cart or pole termination boundaries;
- observation noise;
- one-step action delay;
- dropped or stale observations.

These require a small wrapper or controlled state-initialization harness.
Initially use them only for final testing, then optionally include them in the
evolution objective.

### Suite 5: Physics robustness

Finally vary simulator parameters such as:

- force magnitude;
- gravity;
- cart mass;
- pole mass;
- pole length.

Physics variation is the most expensive and least necessary stage for the
first result, so it should be added only after the earlier suites are passing.

The progression is:

```text
Suite 1: nominal correctness
    -> Suite 2: lower evaluation noise
    -> Suite 3: wider starts
    -> Suite 4: sensor and recovery robustness
    -> Suite 5: model mismatch robustness
```

## Fitness

The evaluator must return a scalar `combined_score` in `[0, 1]`, plus detailed
metrics. Complexity should be reported explicitly and included as a small
penalty rather than hidden inside the rollout metrics.

```text
mean_survival      = mean(episode_length) / 500
worst_survival     = min(episode_length) / 500
cvar_survival      = mean(bottom 20% episode lengths) / 500
success_rate       = fraction of episodes reaching 500 steps
complexity_score    = 1 - parameter_count / MAX_PARAMETERS
complexity_penalty  = 1 - complexity_score

combined_score = 0.50 * mean_survival
               + 0.25 * worst_survival
               + 0.20 * cvar_survival
               + 0.05 * success_rate
               - 0.05 * complexity_penalty
```

The score should be clipped to `[0, 1]` after applying the penalty. The
complexity weight is intentionally small: a controller with more parameters
should be allowed to win if it produces a meaningful robustness improvement.
The penalty primarily prevents bloated parameter schemas from winning by
small, noisy margins.

The evaluator should return at least these metrics:

```text
combined_score
mean_survival
median_survival
worst_case_survival
cvar_survival
success_rate
action_switch_rate
parameter_count
complexity_score
complexity_penalty
```

`parameter_count` is the number of entries returned by `parameter_spec()`.
`complexity_score` is a normalized diagnostic feature, while
`complexity_penalty` is the value subtracted from the task score. If a
parameter-free controller is valid, it receives a complexity score of `1.0`
and a penalty of `0.0`.

For example, with `MAX_PARAMETERS = 16` and five declared parameters:

```text
complexity_score   = 1 - 5 / 16 = 0.6875
complexity_penalty = 0.3125
penalty contribution = 0.05 * 0.3125 = 0.015625
```

This penalty should not be used as a MAP-Elites feature at the same time as
`parameter_count`; that would duplicate the same diversity signal. Use
`parameter_count` only as a diagnostic initially, and reserve the MAP-Elites
features for behavior:

```yaml
feature_dimensions:
  - robust_survival
  - action_switch_rate
```

The primary objective should not be only mean reward. CartPole's default
reward is one per step, so that objective is equivalent to survival time and
can overvalue controllers that succeed only on lucky seeds.

## Artifacts and visualizations

For each evaluated candidate, return compact artifacts such as:

```text
best_params.json
scenario_summary.json
episode_lengths.json
failure_reasons.json
representative_trajectory.json
```

Only selected candidates should produce rendered media. `visualize.py` should
support:

- CartPole GIF or MP4 rollout;
- cart position and pole angle over time;
- pole angle versus angular velocity phase plot;
- action sequence and action-switch rate;
- score distributions by scenario;
- comparison of PID, tuned PID, and evolved controllers;
- best score versus evolution iteration.

The current OpenEvolve visualizer can show evolution lineage and metrics, but
it does not understand CartPole trajectories or render videos. The first
version should therefore use a standalone analysis script. A future framework
feature could add typed artifacts and domain-specific visualization plugins.

## OpenEvolve configuration direction

The example should use:

- cascade evaluation, with cheap validation before full rollouts;
- fixed random seed configuration;
- evaluator artifacts enabled;
- multiple islands for controller-structure diversity;
- a modest per-candidate tuning budget;
- full rewrites initially, since the controller is small and its API should
  remain stable.

The prompt should explicitly tell the LLM that it is evolving controller
logic, not tuning numeric parameters, and that the evaluator owns the
parameter vector.

## Implementation order

1. Implement and test the standalone CartPole rollout harness.
2. Implement the PID baseline and tune it numerically.
3. Add deterministic metrics and scenario artifacts.
4. Add `initial_program.py`, `evaluator.py`, and `config.yaml`.
5. Run a short OpenEvolve experiment.
6. Add `visualize.py` and compare baseline/evolved behavior.
7. Add held-out robustness tests.

Only after this example is working should we extract reusable framework
features such as a scenario abstraction, a numerical-tuning interface, typed
trajectory artifacts, or evaluation-budget accounting.
