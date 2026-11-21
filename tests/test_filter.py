import unittest
from unittest.mock import patch
import numpy as np

from tracking import Estimate, predict, correct, transition, tracking_step
from tracking.experiment import (
    simulate,
    run_filter,
    CONFIGURATIONS,
    MEASUREMENT_COV,
    PRIOR_COV,
    PRIOR_MEAN,
)


def batch_oracle(
    prior, matrix, process_cov, measurement_matrix, measurement_cov, observations
):
    """Independent joint-Gaussian least squares; no sequential filter calls."""
    size = len(prior.mean)
    total = size * (len(observations) + 1)
    rows, values = [], []
    constraint = np.zeros((size, total))
    constraint[:, :size] = np.eye(size)
    chol = np.linalg.cholesky(prior.covariance)
    rows.append(np.linalg.solve(chol, constraint))
    values.append(np.linalg.solve(chol, prior.mean))
    for step, observation in enumerate(observations, 1):
        constraint = np.zeros((size, total))
        constraint[:, size * (step - 1) : size * step] = -matrix
        constraint[:, size * step : size * (step + 1)] = np.eye(size)
        chol = np.linalg.cholesky(process_cov)
        rows.append(np.linalg.solve(chol, constraint))
        values.append(np.zeros(size))
        if observation is not None:
            dimension = len(observation)
            constraint = np.zeros((dimension, total))
            constraint[:, size * step : size * (step + 1)] = measurement_matrix
            chol = np.linalg.cholesky(measurement_cov)
            rows.append(np.linalg.solve(chol, constraint))
            values.append(np.linalg.solve(chol, observation))
    design, response = np.vstack(rows), np.concatenate(values)
    means = np.linalg.lstsq(design, response, rcond=None)[0]
    # A=QR, information=R^T R. Solve only the last-state covariance block.
    _, triangular = np.linalg.qr(design, mode="reduced")
    selector = np.zeros((size, total))
    selector[:, -size:] = np.eye(size)
    whitened_selector = np.linalg.solve(triangular.T, selector.T)
    return means[-size:], whitened_selector.T @ whitened_selector


class FilterTests(unittest.TestCase):
    def test_analytic_scalar_predict_and_joseph_update(self):
        initial = Estimate([2], [[4]])
        predicted = predict(initial, [[1]], [[2]])
        self.assertEqual(predicted.covariance[0, 0], 6)
        posterior, innovation = correct(initial, [5], [[1]], [[9]])
        self.assertAlmostEqual(posterior.mean[0], 38 / 13)
        self.assertAlmostEqual(posterior.covariance[0, 0], 36 / 13)
        self.assertAlmostEqual(innovation.nis, 9 / 13)

    def test_24_independent_batch_gaussian_oracles_with_missing_observations(self):
        rng = np.random.default_rng(78491)
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        for case in range(24):
            dt = float(rng.uniform(0.1, 1.5))
            q = float(rng.uniform(0.02, 1.5))
            matrix, noise = transition(dt, q)
            root = rng.normal(size=(4, 4))
            prior = Estimate(rng.normal(size=4), root @ root.T + np.eye(4) * 0.5)
            observations = [
                None if step % 3 == case % 3 else rng.normal(size=2)
                for step in range(5)
            ]
            current = prior
            with patch(
                "numpy.linalg.inv",
                side_effect=AssertionError("explicit inverse forbidden"),
            ):
                for observed in observations:
                    predicted = predict(current, matrix, noise)
                    current = (
                        predicted
                        if observed is None
                        else correct(predicted, observed, h, MEASUREMENT_COV)[0]
                    )
                mean, cov = batch_oracle(
                    prior, matrix, noise, h, MEASUREMENT_COV, observations
                )
            np.testing.assert_allclose(current.mean, mean, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(current.covariance, cov, rtol=1e-9, atol=1e-9)

    def test_continuous_noise_convention_and_composition(self):
        matrix, noise = transition(2.0, 3.0)
        expected = np.array([[8, 0, 6, 0], [0, 8, 0, 6], [6, 0, 6, 0], [0, 6, 0, 6]])
        np.testing.assert_array_equal(noise, expected)
        first, q_first = transition(0.3, 0.7)
        second, q_second = transition(0.8, 0.7)
        combined, q_combined = transition(1.1, 0.7)
        np.testing.assert_allclose(second @ first, combined, atol=1e-15)
        np.testing.assert_allclose(
            second @ q_first @ second.T + q_second, q_combined, atol=1e-15
        )
        np.testing.assert_array_equal(transition(0.2, 0)[1], np.zeros((4, 4)))

    def test_missing_observation_and_inclusive_gate(self):
        initial = Estimate(PRIOR_MEAN, PRIOR_COV)
        predicted, posterior, innovation = tracking_step(
            initial, None, 0.5, 0.18, MEASUREMENT_COV
        )
        self.assertIs(predicted, posterior)
        self.assertIsNone(innovation)
        initial = Estimate([0], [[1]])
        accepted, detail = correct(initial, [2], [[1]], [[3]], gate=1.0)
        self.assertTrue(detail.accepted)
        rejected, detail = correct(initial, [2], [[1]], [[3]], gate=0.99)
        self.assertFalse(detail.accepted)
        self.assertIs(rejected, initial)
        self.assertEqual(detail.nis, 1)
        self.assertEqual(accepted.mean[0], 0.5)

    def test_covariance_psd_symmetry_and_reproducible_records(self):
        data = simulate(38512)
        self.assertEqual(data, simulate(38512))
        for label in CONFIGURATIONS:
            result = run_filter(data, label)
            self.assertEqual(result, run_filter(data, label))
            for record in result["records"]:
                cov = np.asarray(record["posterior"]["covariance"])
                np.testing.assert_allclose(cov, cov.T, atol=1e-14)
                self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)
        initial = Estimate(np.zeros(4), np.eye(4))
        for _ in range(200):
            initial = tracking_step(
                initial, [1.0, 2.0], 0.01, 0.0, np.diag([1e-8, 1e-6])
            )[1]
        self.assertGreaterEqual(np.linalg.eigvalsh(initial.covariance).min(), -1e-14)

    def test_validation_and_defensive_arrays(self):
        mean, cov = np.zeros(2), np.eye(2)
        initial = Estimate(mean, cov)
        mean[0] = 8
        cov[0, 0] = 10
        self.assertEqual(initial.mean[0], 0)
        self.assertEqual(initial.covariance[0, 0], 1)
        with self.assertRaises(ValueError):
            initial.mean[0] = 2
        for bad in ([[1, 2], [0, 1]], [[1, 0], [0, -1]], [[1, 0], [0, float("nan")]]):
            with self.assertRaises(ValueError):
                Estimate([0, 0], bad)
        for dt, q in ((0, 1), (True, 0.1), (1, -0.1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                transition(dt, q)
        with self.assertRaises(ValueError):
            correct(initial, [1], [[1, 0]], [[0]])
        with self.assertRaises(ValueError):
            correct(initial, [float("nan")], [[1, 0]], [[1]])
        with self.assertRaises(ValueError):
            correct(initial, [1], [[1, 0]], [[1]], gate=float("nan"))

    def test_experiment_result_does_not_alias_configuration_or_truth(self):
        data = simulate(71, 2)
        result = run_filter(data, "matched")
        original_q = CONFIGURATIONS["matched"]["q"]
        result["settings"]["q"] = 80
        result["records"][0]["truth"][0] = -999
        self.assertEqual(CONFIGURATIONS["matched"]["q"], original_q)
        self.assertNotEqual(data["truth"][0][0], -999)
        self.assertEqual(run_filter(data, "matched")["settings"]["q"], original_q)


if __name__ == "__main__":
    unittest.main()
