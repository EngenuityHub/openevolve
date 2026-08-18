"""Initial CartPole controller for OpenEvolve.

Only the controller implementation is intended to evolve. Numeric values are
provided by the evaluator after numerical tuning.
"""

# EVOLVE-BLOCK-START


def parameter_spec():
    return [
        {"name": "kp_x", "low": -10.0, "high": 10.0},
        {"name": "kd_x", "low": -10.0, "high": 10.0},
        {"name": "kp_theta", "low": -30.0, "high": 30.0},
        {"name": "kd_theta", "low": -10.0, "high": 10.0},
        {"name": "threshold", "low": -2.0, "high": 2.0},
    ]


class Controller:
    def __init__(self, params):
        self.params = params

    def reset(self):
        pass

    def act(self, observation):
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        p = self.params
        control = (
            p["kp_x"] * cart_position
            + p["kd_x"] * cart_velocity
            + p["kp_theta"] * pole_angle
            + p["kd_theta"] * pole_angular_velocity
        )
        return int(control >= p["threshold"])


def make_controller(params):
    return Controller(params)

# EVOLVE-BLOCK-END
