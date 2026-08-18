"""Shared candidate loading, numerical tuning, and rollout utilities."""

import importlib.util
import json
import math
import uuid

import gymnasium as gym
import numpy as np
from scipy.optimize import differential_evolution

from scenarios import MAX_STEPS, TUNING_SEEDS, nominal_scenario

MAX_PARAMETERS = 16
MAX_TUNING_EVALUATIONS = 250
GLOBAL_ABS_PARAMETER_LIMIT = 100.0


def load_candidate(program_path):
    module_name = f"cartpole_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_controller"):
        raise ValueError("Candidate must define make_controller(params)")
    if not hasattr(module, "parameter_spec"):
        raise ValueError("Candidate must define parameter_spec()")
    return module


def validated_parameter_spec(candidate):
    raw_spec = candidate.parameter_spec()
    if not isinstance(raw_spec, list) or len(raw_spec) > MAX_PARAMETERS:
        raise ValueError(f"parameter_spec() must return at most {MAX_PARAMETERS} entries")

    result = []
    names = set()
    for item in raw_spec:
        if not isinstance(item, dict):
            raise ValueError("Each parameter specification must be a dictionary")
        name = item.get("name")
        low = item.get("low")
        high = item.get("high")
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError(f"Invalid or duplicate parameter name: {name!r}")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"Bounds for {name!r} must be numeric")
        if not math.isfinite(float(low)) or not math.isfinite(float(high)):
            raise ValueError(f"Bounds for {name!r} must be finite")
        if not low < high or max(abs(float(low)), abs(float(high))) > GLOBAL_ABS_PARAMETER_LIMIT:
            raise ValueError(f"Unsafe bounds for {name!r}: {low}, {high}")
        names.add(name)
        result.append({"name": name, "low": float(low), "high": float(high)})
    return result


def params_from_vector(spec, vector):
    return {item["name"]: float(value) for item, value in zip(spec, vector)}


def run_episode(candidate, params, scenario, capture=False, render=False):
    render_mode = "rgb_array" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    controller = candidate.make_controller(dict(params))
    if not hasattr(controller, "act"):
        env.close()
        raise ValueError("make_controller(params) must return an object with act()")
    reset = getattr(controller, "reset", None)
    if reset is not None:
        reset()

    observation, _ = env.reset(
        seed=int(scenario["seed"]),
        options={"low": scenario["low"], "high": scenario["high"]},
    )
    frames = []
    observations = [np.asarray(observation, dtype=float).tolist()]
    actions = []
    rewards = []
    terminated = truncated = False
    try:
        for _ in range(int(scenario.get("max_steps", MAX_STEPS))):
            action = controller.act(observation)
            if isinstance(action, np.ndarray):
                action = action.item()
            if int(action) not in (0, 1):
                raise ValueError(f"Controller returned invalid action: {action!r}")
            action = int(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            observations.append(np.asarray(observation, dtype=float).tolist())
            if capture:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "length": len(actions),
        "success": bool(truncated and not terminated and len(actions) >= scenario.get("max_steps", MAX_STEPS)),
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "frames": frames,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def tune_candidate(candidate, seeds=TUNING_SEEDS, maxiter=8, popsize=4):
    spec = validated_parameter_spec(candidate)
    bounds = [(item["low"], item["high"]) for item in spec]
    calls = 0

    def objective(vector):
        nonlocal calls
        calls += 1
        params = params_from_vector(spec, vector)
        lengths = []
        try:
            for seed in seeds:
                result = run_episode(candidate, params, nominal_scenario(seed))
                lengths.append(result["length"] / MAX_STEPS)
        except Exception:
            return 1.0
        return -float(np.mean(lengths))

    if not spec:
        params = {}
        return params, 0.0, calls

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=0,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="immediate",
    )
    if calls > MAX_TUNING_EVALUATIONS:
        raise RuntimeError("Tuning evaluation budget exceeded")
    return params_from_vector(spec, result.x), float(-result.fun), calls


def jsonable_params(params):
    return json.dumps({key: float(value) for key, value in params.items()}, sort_keys=True)
