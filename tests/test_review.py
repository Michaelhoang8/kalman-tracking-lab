"""Coordinate-change checks independent of the trajectory generator."""
import unittest
import numpy as np

from tracking.filter import Estimate, predict, correct


class ReviewTests(unittest.TestCase):
    def test_nonorthogonal_state_basis_preserves_innovations_and_posterior(self):
        mean = np.array([1.5, -0.7])
        prior = np.array([[4., 1.2], [1.2, 2.]])
        transition = np.array([[1., .3], [0., 1.]])
        process = np.array([[.2, .04], [.04, .1]])
        observation_matrix = np.array([[1., -.4]])
        measurement = np.array([[.7]])
        basis = np.array([[2., 1.], [-.5, 3.]])
        transformed_transition = np.linalg.solve(basis.T, (basis @ transition).T).T
        transformed_measurement = np.linalg.solve(basis.T, observation_matrix.T).T
        predicted = predict(Estimate(mean, prior), transition, process)
        transformed = predict(Estimate(basis @ mean, basis @ prior @ basis.T),
                              transformed_transition, basis @ process @ basis.T)
        for gate in [None, .001]:
            posterior, innovation = correct(predicted, [2.4], observation_matrix, measurement, gate)
            other, other_innovation = correct(transformed, [2.4], transformed_measurement, measurement, gate)
            np.testing.assert_allclose(other.mean, basis @ posterior.mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(other.covariance, basis @ posterior.covariance @ basis.T, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(other_innovation.residual, innovation.residual, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(other_innovation.covariance, innovation.covariance, rtol=1e-12, atol=1e-12)
            self.assertAlmostEqual(other_innovation.nis, innovation.nis, places=12)
            self.assertEqual(other_innovation.accepted, innovation.accepted)


if __name__ == '__main__':
    unittest.main()
