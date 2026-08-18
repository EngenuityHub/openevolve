"""OpenEvolve evaluator for the first CartPole scenario suite."""

import json
import time

import numpy as np

from openevolve.evaluation_result import EvaluationResult

from cartpole_common import (
    MAX_PARAMETERS,
    MAX_STEPS,
    jsonable_params,
    load_candidate,
    run_episode,
    tune_candidate,
)
from scenarios import validation_scenarios


def _score(candidate, params):
    episodes = []
    switch_count = 0
    action_count = 0
    for scenario in validation_scenarios():
        episode = run_episode(candidate, params, scenario)
        actions = episode["actions"]
        switch_count += sum(a != b for a, b in zip(actions, actions[1:]))
        action_count += max(0, len(actions) - 1)
        episodes.append({
            "scenario": scenario["name"],
            "seed": scenario["seed"],
            "length": episode["length"],
            "success": episode["success"],
        })

    lengths = np.asarray([item["length"] for item in episodes], dtype=float)
    normalized = lengths / MAX_STEPS
    tail_count = max(1, int(np.ceil(0.2 * len(normalized))))
    cvar = float(np.mean(np.sort(normalized)[:tail_count]))
    spec_count = len(candidate.parameter_spec())
    complexity_penalty = min(1.0, spec_count / 16.0)
    task_score = (
        0.50 * float(np.mean(normalized))
        + 0.25 * float(np.min(normalized))
        + 0.20 * cvar
        + 0.05 * float(np.mean([item["success"] for item in episodes]))
    )
    combined_score = float(np.clip(task_score - 0.05 * complexity_penalty, 0.0, 1.0))

    metrics = {
        "combined_score": combined_score,
        "mean_survival": float(np.mean(normalized)),
        "median_survival": float(np.median(normalized)),
        "worst_case_survival": float(np.min(normalized)),
        "cvar_survival": cvar,
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "action_switch_rate": float(switch_count / action_count) if action_count else 0.0,
        "parameter_count": float(spec_count),
        "complexity_score": float(1.0 - complexity_penalty),
        "complexity_penalty": float(complexity_penalty),
    }
    return metrics, episodes


def evaluate(program_path):
    started = time.time()
    try:
        candidate = load_candidate(program_path)
        params, tuning_score, tuning_calls = tune_candidate(candidate)
        metrics, episodes = _score(candidate, params)
        metrics.update({
            "tuning_score": float(tuning_score),
            "tuning_evaluations": float(tuning_calls),
            "evaluation_time_seconds": float(time.time() - started),
        })
        artifacts = {
            "tuned_params": jsonable_params(params),
            "scenario_summary": json.dumps(episodes, sort_keys=True),
            "evaluation_suite": "suite_1_nominal",
        }
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        return EvaluationResult(
            # Keep configured MAP-Elites dimensions present even when an
            # evolved candidate is invalid, so the framework can score the
            # failure cleanly instead of failing feature extraction.
            metrics={
                "combined_score": 0.0,
                "mean_survival": 0.0,
                "action_switch_rate": 0.0,
                "parameter_count": float(MAX_PARAMETERS),
                "complexity_score": 0.0,
                "complexity_penalty": 1.0,
                "evaluation_error": 1.0,
            },
            artifacts={"error": repr(exc)},
        )


def evaluate_stage1(program_path):
    """Cheap interface/schema validation for cascade evaluation."""
    try:
        candidate = load_candidate(program_path)
        spec = candidate.parameter_spec()
        if not isinstance(spec, list) or len(spec) > 16:
            raise ValueError("Invalid parameter schema")
        return {"combined_score": 1.0, "stage1_passed": 1.0}
    except Exception as exc:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "stage1_passed": 0.0},
            artifacts={"stage1_error": repr(exc)},
        )


def evaluate_stage2(program_path):
    """Full tuned evaluation after the cheap interface check."""
    return evaluate(program_path)
