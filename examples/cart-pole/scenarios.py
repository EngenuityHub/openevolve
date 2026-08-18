"""Small, deterministic CartPole scenario suite for the first experiment."""

TUNING_SEEDS = (0, 1, 2, 3, 4)
VALIDATION_SEEDS = (100, 101, 102, 103, 104, 105, 106, 107, 108, 109)
DEFAULT_LOW = -0.05
DEFAULT_HIGH = 0.05
MAX_STEPS = 500


def nominal_scenario(seed):
    return {
        "name": "nominal",
        "seed": int(seed),
        "low": DEFAULT_LOW,
        "high": DEFAULT_HIGH,
        "max_steps": MAX_STEPS,
    }


def validation_scenarios():
    return [nominal_scenario(seed) for seed in VALIDATION_SEEDS]
