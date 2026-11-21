"""Seeded model-consistent trajectories and full per-step filter records."""

import numpy as np
from .filter import Estimate, transition, tracking_step

PRIOR_MEAN = np.array([0.0, 0.0, 1.0, 0.35])
PRIOR_COV = np.diag([4.0, 4.0, 1.0, 1.0])
MEASUREMENT_COV = np.array([[2.25, 0.45], [0.45, 1.44]])
TRUE_Q = 0.18
DT = 0.5
STEPS = 60
GATE = float(-2 * np.log(0.01))  # Chi-square(df=2) 99% threshold.
POSITION_95 = float(-2 * np.log(0.05))  # Chi-square(df=2) 95% ellipse.

CONFIGURATIONS = {
    "matched": {"q": TRUE_Q, "schedule": "all", "gate": None, "contamination": False},
    "low_process_noise": {
        "q": TRUE_Q / 30,
        "schedule": "all",
        "gate": None,
        "contamination": False,
    },
    "sparse_measurements": {
        "q": TRUE_Q,
        "schedule": "sparse",
        "gate": None,
        "contamination": False,
    },
    "outliers_ungated": {
        "q": TRUE_Q,
        "schedule": "all",
        "gate": None,
        "contamination": True,
    },
    "outliers_gated": {
        "q": TRUE_Q,
        "schedule": "all",
        "gate": GATE,
        "contamination": True,
    },
}


def simulate(seed: int, steps: int = STEPS) -> dict:
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(steps) is not int
        or not 1 <= steps <= 200
    ):
        raise ValueError("invalid seed or step count")
    rng = np.random.default_rng(seed)
    matrix, noise = transition(DT, TRUE_Q)
    truth = [rng.multivariate_normal(PRIOR_MEAN, PRIOR_COV)]
    observed = []
    for _ in range(steps):
        truth.append(matrix @ truth[-1] + rng.multivariate_normal(np.zeros(4), noise))
        observed.append(
            truth[-1][:2] + rng.multivariate_normal(np.zeros(2), MEASUREMENT_COV)
        )
    return {
        "seed": seed,
        "dt": DT,
        "truth": np.asarray(truth).tolist(),
        "observations": np.asarray(observed).tolist(),
    }


def run_filter(data: dict, label: str) -> dict:
    config = dict(CONFIGURATIONS[label])
    estimate = Estimate(PRIOR_MEAN, PRIOR_COV)
    records = [
        {
            "step": 0,
            "time": 0.0,
            "truth": list(data["truth"][0]),
            "posterior": estimate.to_dict(),
            "predicted": None,
            "observation": None,
            "innovation": None,
            "injected_outlier": False,
        }
    ]
    for index, clean in enumerate(data["observations"], 1):
        observation = np.array(clean, dtype=float)
        outlier = config["contamination"] and index in (18, 36, 48)
        if outlier:
            # Fixed, disclosed additive faults; same fault in the gated/ungated pair.
            observation += np.array([11.0, -9.0]) * (1 if index != 36 else -1)
        if config["schedule"] == "sparse" and (index % 3 != 0 or 25 <= index <= 36):
            observation = None
        predicted, estimate, innovation = tracking_step(
            estimate,
            observation,
            data["dt"],
            config["q"],
            MEASUREMENT_COV,
            config["gate"],
        )
        records.append(
            {
                "step": index,
                "time": index * data["dt"],
                "truth": list(data["truth"][index]),
                "predicted": predicted.to_dict(),
                "posterior": estimate.to_dict(),
                "observation": None if observation is None else observation.tolist(),
                "innovation": None if innovation is None else innovation.to_dict(),
                "injected_outlier": outlier,
            }
        )
    return {
        "configuration": label,
        "settings": config,
        "seed": data["seed"],
        "records": records,
    }


def final_metrics(result: dict) -> dict:
    final = result["records"][-1]
    error = np.array(final["truth"]) - np.array(final["posterior"]["mean"])
    cov = np.array(final["posterior"]["covariance"])
    nees = float(error @ np.linalg.solve(cov, error))
    position_nees = float(error[:2] @ np.linalg.solve(cov[:2, :2], error[:2]))
    innovation = final["innovation"]
    return {
        "position_squared_error": float(error[:2] @ error[:2]),
        "velocity_squared_error": float(error[2:] @ error[2:]),
        "nees": nees,
        "position_nees": position_nees,
        "position_inside_95": position_nees <= POSITION_95,
        "final_nis": None if innovation is None else innovation["nis"],
        "rejected_observations": sum(
            record["innovation"] is not None and not record["innovation"]["accepted"]
            for record in result["records"]
        ),
    }
