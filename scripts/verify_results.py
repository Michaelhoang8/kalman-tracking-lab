"""Validate full saved arrays and recompute fixed-time statistics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.experiment import (
    CONFIGURATIONS,
    run_filter,
    final_metrics,
    STEPS,
    DT,
    POSITION_95,
    PRIOR_MEAN,
    PRIOR_COV,
    MEASUREMENT_COV,
)
from scripts.run_study import sources, summarize


def verify_artifacts(out: Path):
    provenance = json.loads((out / "results/provenance.json").read_text())
    assert provenance["source_sha256_lf"] == sources()
    for name, digest in provenance["artifact_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
    schema = json.loads((out / "results/archive-schema.json").read_text())
    summary = json.loads((out / "results/summary.json").read_text())
    metrics = json.loads((out / "results/final-metrics.json").read_text())
    # NPZ members are lazily decompressed on access. Load each member once;
    # otherwise per-step indexing repeatedly decompresses entire history arrays.
    with np.load(out / "results/monte-carlo.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    assert set(data) == set(schema["arrays"])
    for name, expected in schema["arrays"].items():
        assert list(data[name].shape) == expected["shape"]
        assert str(data[name].dtype) == expected["dtype"]
    seeds = data["seeds"]
    assert len(set(seeds.tolist())) == len(seeds)
    labels = list(CONFIGURATIONS)
    assert schema["configuration_axis"] == labels
    mean, cov = data["posterior_mean"], data["posterior_covariance"]
    assert mean.shape == (5, len(seeds), STEPS + 1, 4)
    assert np.isfinite(mean).all() and np.isfinite(cov).all()
    np.testing.assert_allclose(cov, cov.swapaxes(-1, -2), rtol=0, atol=1e-11)
    assert np.linalg.eigvalsh(cov).min() >= -1e-10
    assert not data["observed"][:, :, 0].any()
    assert np.isnan(data["predicted_mean"][:, :, 0]).all()
    for index, label in enumerate(labels):
        config = CONFIGURATIONS[label]
        recalculated = []
        for run, seed in enumerate(seeds):
            np.testing.assert_array_equal(mean[index, run, 0], PRIOR_MEAN)
            np.testing.assert_array_equal(cov[index, run, 0], PRIOR_COV)
            for step in range(1, STEPS + 1):
                # Independently write F and integrated Q; do not call filter helpers.
                f = np.eye(4)
                f[0, 2] = f[1, 3] = DT
                q = np.zeros((4, 4))
                for axis in (0, 1):
                    q[axis, axis] = config["q"] * DT**3 / 3
                    q[axis + 2, axis + 2] = config["q"] * DT
                    q[axis, axis + 2] = q[axis + 2, axis] = config["q"] * DT**2 / 2
                predicted_mean = f @ mean[index, run, step - 1]
                predicted_cov = f @ cov[index, run, step - 1] @ f.T + q
                np.testing.assert_allclose(
                    predicted_mean, data["predicted_mean"][index, run, step], atol=1e-10
                )
                np.testing.assert_allclose(
                    predicted_cov,
                    data["predicted_covariance"][index, run, step],
                    atol=1e-10,
                )
                observed = config["schedule"] == "all" or (
                    step % 3 == 0 and not 25 <= step <= 36
                )
                assert observed == data["observed"][index, run, step]
                fault = config["contamination"] and step in (18, 36, 48)
                assert fault == data["injected_outlier"][index, run, step]
                if observed:
                    observation = data["clean_observations"][run, step - 1].copy()
                    if fault:
                        observation += np.array([11.0, -9.0]) * (
                            1 if step != 36 else -1
                        )
                    np.testing.assert_allclose(
                        observation,
                        data["used_observation"][index, run, step],
                        atol=1e-12,
                    )
                    residual = observation - predicted_mean[:2]
                    innovation_cov = predicted_cov[:2, :2] + MEASUREMENT_COV
                    nis = float(residual @ np.linalg.solve(innovation_cov, residual))
                    np.testing.assert_allclose(
                        residual,
                        data["innovation_residual"][index, run, step],
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        innovation_cov,
                        data["innovation_covariance"][index, run, step],
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        nis, data["nis"][index, run, step], atol=1e-9
                    )
                    accepted = config["gate"] is None or nis <= config["gate"]
                    assert accepted == data["accepted"][index, run, step]
                    if accepted:
                        gain = np.linalg.solve(innovation_cov, predicted_cov[:2, :]).T
                        expected_mean = predicted_mean + gain @ residual
                        # Independent conditional-Gaussian covariance, not Joseph form.
                        expected_cov = predicted_cov - gain @ innovation_cov @ gain.T
                    else:
                        expected_mean, expected_cov = predicted_mean, predicted_cov
                else:
                    assert not data["accepted"][index, run, step]
                    assert np.isnan(data["nis"][index, run, step])
                    expected_mean, expected_cov = predicted_mean, predicted_cov
                np.testing.assert_allclose(
                    mean[index, run, step], expected_mean, atol=1e-9
                )
                np.testing.assert_allclose(
                    cov[index, run, step], expected_cov, atol=1e-9
                )
            error = data["truth"][run, -1] - mean[index, run, -1]
            final_cov = cov[index, run, -1]
            nees = float(error @ np.linalg.solve(final_cov, error))
            position_nees = float(
                error[:2] @ np.linalg.solve(final_cov[:2, :2], error[:2])
            )
            row = {
                "seed": int(seed),
                "position_squared_error": float(error[:2] @ error[:2]),
                "velocity_squared_error": float(error[2:] @ error[2:]),
                "nees": nees,
                "position_nees": position_nees,
                "position_inside_95": position_nees <= POSITION_95,
                "final_nis": float(data["nis"][index, run, -1]),
                "rejected_observations": int(
                    np.sum(data["observed"][index, run] & ~data["accepted"][index, run])
                ),
            }
            for key, value in row.items():
                if isinstance(value, float):
                    np.testing.assert_allclose(
                        value, metrics[label][run][key], atol=1e-9
                    )
                else:
                    assert value == metrics[label][run][key]
            recalculated.append(row)
        checked = summarize(label, recalculated)
        for key, value in checked.items():
            if isinstance(value, (float, list)):
                np.testing.assert_allclose(value, summary[index][key], atol=1e-10)
            else:
                assert value == summary[index][key]
    illustrative = json.loads((out / "results/illustrative-input.json").read_text())
    for label in labels:
        saved = json.loads((out / "results/illustrative" / f"{label}.json").read_text())
        rerun = run_filter(illustrative, label)
        for first, second in zip(saved["records"], rerun["records"]):
            np.testing.assert_allclose(
                first["posterior"]["mean"], second["posterior"]["mean"], atol=1e-9
            )
            np.testing.assert_allclose(
                first["posterior"]["covariance"],
                second["posterior"]["covariance"],
                atol=1e-9,
            )
        final_metrics(saved)
    with (out / "results/timings.csv").open(newline="", encoding="utf-8") as handle:
        timing = list(csv.DictReader(handle))
    env = json.loads((out / "results/environment.json").read_text())
    assert len(timing) == env["timing"]["samples_per_configuration"] * 5
    for index, row in enumerate(timing, 1):
        assert int(row["ordinal"]) == index and int(row["elapsed_ns"]) > 0
        assert int(row["steps"]) == STEPS and row["configuration"] in labels
    print(
        f"Verified all {5 * len(seeds) * STEPS:,} estimated steps, covariance checks, fixed-time summaries and hashes."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    verify_artifacts(parser.parse_args().output_dir.resolve())
