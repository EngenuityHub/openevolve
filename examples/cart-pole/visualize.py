"""Plot evolution progress and render a selected CartPole controller.

Examples:
  uv run python visualize.py progress --output-dir openevolve_output
  uv run python visualize.py rollout --program openevolve_output/best/best_program.py
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from cartpole_common import jsonable_params, load_candidate, run_episode, tune_candidate
from scenarios import nominal_scenario


def progress(output_dir, destination):
    rows = []
    for info_path in sorted(Path(output_dir).glob("checkpoints/checkpoint_*/best_program_info.json")):
        with info_path.open() as handle:
            info = json.load(handle)
        row = {
            "iteration": info.get(
                "current_iteration",
                info.get("iteration", info.get("generation", 0)),
            )
        }
        row.update(info.get("metrics", {}))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No checkpoint best_program_info.json files in {output_dir}")

    rows.sort(key=lambda row: row["iteration"])
    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(iterations, [row.get("combined_score", 0.0) for row in rows], marker="o", label="combined")
    axes[0].plot(iterations, [row.get("mean_survival", 0.0) for row in rows], marker=".", label="mean survival")
    axes[0].plot(iterations, [row.get("worst_case_survival", 0.0) for row in rows], marker=".", label="worst survival")
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(iterations, [row.get("complexity_penalty", 0.0) for row in rows], marker="o", label="complexity penalty")
    axes[1].plot(iterations, [row.get("action_switch_rate", 0.0) for row in rows], marker=".", label="action switch rate")
    axes[1].set_xlabel("evolution iteration")
    axes[1].set_ylabel("diagnostic")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    print(f"Wrote {destination}")


def rollout(program_path, destination, seed, low, high, params_path=None):
    candidate = load_candidate(program_path)
    if params_path:
        with open(params_path) as handle:
            params = json.load(handle)
    else:
        # Match the bounded Suite 1 tuning budget. A later final-retuning
        # command can intentionally use a larger, separately configured budget.
        params, _, _ = tune_candidate(candidate, maxiter=8, popsize=4)

    scenario = nominal_scenario(seed)
    scenario["low"] = low
    scenario["high"] = high
    result = run_episode(candidate, params, scenario, capture=True, render=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "controller_params.json").write_text(jsonable_params(params))
    if result["frames"]:
        imageio.mimsave(destination / "rollout.gif", result["frames"], duration=1 / 50)

    observations = np.asarray(result["observations"])
    time = np.arange(len(observations))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, observations[:, 0], label="cart position")
    axes[0].plot(time, observations[:, 2], label="pole angle")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time, observations[:, 1], label="cart velocity")
    axes[1].plot(time, observations[:, 3], label="pole angular velocity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].step(np.arange(len(result["actions"])), result["actions"], where="post")
    axes[2].set_ylabel("action")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"CartPole rollout: {result['length']} steps, seed={seed}")
    fig.tight_layout()
    fig.savefig(destination / "rollout_trajectory.png", dpi=140)
    print(f"Controller parameters: {params}")
    print(f"Survival: {result['length']} / {scenario['max_steps']} steps")
    print(f"Wrote visualizations to {destination}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--output-dir", required=True)
    progress_parser.add_argument("--destination", default="evolution_progress.png")
    rollout_parser = subparsers.add_parser("rollout")
    rollout_parser.add_argument("--program", required=True)
    rollout_parser.add_argument("--destination", default="cartpole_rollout")
    rollout_parser.add_argument("--seed", type=int, default=100)
    rollout_parser.add_argument("--low", type=float, default=-0.05)
    rollout_parser.add_argument("--high", type=float, default=0.05)
    rollout_parser.add_argument("--params")
    args = parser.parse_args()
    if args.command == "progress":
        progress(args.output_dir, args.destination)
    else:
        rollout(args.program, args.destination, args.seed, args.low, args.high, args.params)


if __name__ == "__main__":
    main()
