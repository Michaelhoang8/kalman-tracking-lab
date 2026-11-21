"""Small, validated linear Kalman filter; no filtering libraries or matrix inverse."""

from dataclasses import dataclass
import math
import numpy as np


def array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def covariance(value, size: int, label: str, positive: bool = False) -> np.ndarray:
    result = array(value, (size, size), label)
    scale = max(1.0, float(np.max(np.abs(result))))
    if not np.allclose(result, result.T, rtol=0, atol=1e-10 * scale):
        raise ValueError(f"{label} must be symmetric")
    result = (result + result.T) * 0.5
    smallest = float(np.linalg.eigvalsh(result)[0])
    if (positive and smallest <= 0) or smallest < -1e-10 * scale:
        raise ValueError(
            f"{label} must be {'positive definite' if positive else 'positive semidefinite'}"
        )
    return result


def readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class Estimate:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        if mean.ndim != 1 or not 1 <= len(mean) <= 8 or not np.isfinite(mean).all():
            raise ValueError("mean must be a finite vector of length 1..8")
        cov = covariance(self.covariance, len(mean), "covariance")
        object.__setattr__(self, "mean", readonly(mean))
        object.__setattr__(self, "covariance", readonly(cov))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "covariance": self.covariance.tolist()}


@dataclass(frozen=True)
class Innovation:
    residual: np.ndarray
    covariance: np.ndarray
    nis: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "residual": self.residual.tolist(),
            "covariance": self.covariance.tolist(),
            "nis": self.nis,
            "accepted": self.accepted,
        }


def predict(estimate: Estimate, matrix, noise) -> Estimate:
    size = len(estimate.mean)
    matrix = array(matrix, (size, size), "transition")
    noise = covariance(noise, size, "process noise")
    mean = matrix @ estimate.mean
    cov = matrix @ estimate.covariance @ matrix.T + noise
    return Estimate(mean, (cov + cov.T) * 0.5)


def correct(
    predicted: Estimate, observation, matrix, noise, gate: float | None = None
) -> tuple[Estimate, Innovation]:
    size = len(predicted.mean)
    observation = np.array(observation, dtype=np.float64, copy=True)
    if observation.ndim != 1 or not 1 <= len(observation) <= 8:
        raise ValueError("observation must be a vector of length 1..8")
    dimension = len(observation)
    observation = array(observation, (dimension,), "observation")
    matrix = array(matrix, (dimension, size), "measurement matrix")
    noise = covariance(noise, dimension, "measurement noise", positive=True)
    if gate is not None and (
        isinstance(gate, bool)
        or not isinstance(gate, (int, float))
        or not math.isfinite(gate)
        or gate <= 0
    ):
        raise ValueError("gate must be a finite positive threshold or None")
    residual = observation - matrix @ predicted.mean
    innovation_cov = matrix @ predicted.covariance @ matrix.T + noise
    innovation_cov = covariance(
        innovation_cov, dimension, "innovation covariance", positive=True
    )
    # Solve S X = H P; transposing X yields K = P H^T S^-1 without an inverse.
    gain = np.linalg.solve(innovation_cov, matrix @ predicted.covariance).T
    nis = float(residual @ np.linalg.solve(innovation_cov, residual))
    if not math.isfinite(nis):
        raise ValueError("innovation overflowed")
    accepted = gate is None or nis <= gate
    innovation = Innovation(readonly(residual), readonly(innovation_cov), nis, accepted)
    if not accepted:
        return predicted, innovation
    mean = predicted.mean + gain @ residual
    identity_minus = np.eye(size) - gain @ matrix
    # Joseph form retains both transformed prior and measurement covariance.
    cov = (
        identity_minus @ predicted.covariance @ identity_minus.T + gain @ noise @ gain.T
    )
    return Estimate(mean, (cov + cov.T) * 0.5), innovation


def transition(dt: float, spectral_density: float) -> tuple[np.ndarray, np.ndarray]:
    """State [x,y,vx,vy]. Continuous white acceleration intensity q [m²/s³].

    Q = q [[dt³/3 I, dt²/2 I], [dt²/2 I, dt I]].
    This is not the discrete piecewise-constant-acceleration dt⁴/4 convention.
    """
    for value, low, high, label in (
        (dt, 0, 10, "dt"),
        (spectral_density, -1, 100, "spectral_density"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= low
            or value > high
        ):
            raise ValueError(f"invalid {label}")
    if spectral_density < 0:
        raise ValueError("spectral_density must be nonnegative")
    identity = np.eye(2)
    matrix = np.eye(4)
    matrix[:2, 2:] = dt * identity
    noise = spectral_density * np.block(
        [
            [dt**3 / 3 * identity, dt**2 / 2 * identity],
            [dt**2 / 2 * identity, dt * identity],
        ]
    )
    return matrix, noise


def tracking_step(
    estimate: Estimate,
    observation,
    dt: float,
    spectral_density: float,
    measurement_covariance,
    gate: float | None = None,
) -> tuple[Estimate, Estimate, Innovation | None]:
    if len(estimate.mean) != 4:
        raise ValueError("tracking state must have four components")
    matrix, noise = transition(dt, spectral_density)
    predicted = predict(estimate, matrix, noise)
    if observation is None:
        # Validate configured measurement covariance even during a missing step.
        covariance(measurement_covariance, 2, "measurement covariance", positive=True)
        if gate is not None and (
            isinstance(gate, bool)
            or not isinstance(gate, (int, float))
            or not math.isfinite(gate)
            or gate <= 0
        ):
            raise ValueError("gate must be a finite positive threshold or None")
        return predicted, predicted, None
    matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    posterior, innovation = correct(
        predicted, observation, matrix, measurement_covariance, gate
    )
    return predicted, posterior, innovation
