"""Run synthetic Monte Carlo experiments and retain all estimated states."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter_ns
import numpy as np
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.experiment import (
    CONFIGURATIONS,
    simulate,
    run_filter,
    final_metrics,
    STEPS,
)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def write_json(path: Path, value):
    write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sources():
    files = [
        *ROOT.glob("tracking/*.py"),
        *(file for file in ROOT.glob("tests/*.py") if file.name != "test_media.py"),
    ]
    files += [
        ROOT / "scripts" / name
        for name in ("run_study.py", "report.py", "verify_results.py")
    ]
    return {
        file.relative_to(ROOT)
        .as_posix(): hashlib.sha256(file.read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest()
        for file in sorted(files)
    }


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(center - half), float(center + half)]


def summarize(label, rows):
    nees = np.array([row["nees"] for row in rows])
    coverage = sum(row["position_inside_95"] for row in rows)
    return {
        "configuration": label,
        "runs": len(rows),
        "time_index": STEPS,
        "position_rmse": float(
            np.sqrt(np.mean([row["position_squared_error"] for row in rows]))
        ),
        "velocity_rmse": float(
            np.sqrt(np.mean([row["velocity_squared_error"] for row in rows]))
        ),
        "mean_nees": float(nees.mean()),
        "nees_mc_standard_error": float(nees.std(ddof=1) / np.sqrt(len(rows))),
        "position_coverage": coverage / len(rows),
        "coverage_count": coverage,
        "coverage_wilson_95": wilson(coverage, len(rows)),
        "mean_final_pre_gate_nis": float(np.mean([row["final_nis"] for row in rows])),
        "total_rejected": sum(row["rejected_observations"] for row in rows),
    }


def study(out: Path, runs: int, samples: int):
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    write(out / "results/test-results.txt", tests.stdout + tests.stderr)
    print(tests.stdout + tests.stderr)
    tests.check_returncode()
    labels = list(CONFIGURATIONS)
    seed_sequence = np.random.SeedSequence(661024)
    seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(runs)]
    shape = (len(labels), runs, STEPS + 1)
    arrays = {
        "seeds": np.asarray(seeds, dtype=np.uint32),
        "truth": np.zeros((runs, STEPS + 1, 4)),
        "clean_observations": np.zeros((runs, STEPS, 2)),
        "posterior_mean": np.zeros((*shape, 4)),
        "posterior_covariance": np.zeros((*shape, 4, 4)),
        "predicted_mean": np.full((*shape, 4), np.nan),
        "predicted_covariance": np.full((*shape, 4, 4), np.nan),
        "used_observation": np.full((*shape, 2), np.nan),
        "innovation_residual": np.full((*shape, 2), np.nan),
        "innovation_covariance": np.full((*shape, 2, 2), np.nan),
        "nis": np.full(shape, np.nan),
        "observed": np.zeros(shape, dtype=bool),
        "accepted": np.zeros(shape, dtype=bool),
        "injected_outlier": np.zeros(shape, dtype=bool),
    }
    metrics = {label: [] for label in labels}
    for run, seed in enumerate(seeds):
        data = simulate(seed)
        arrays["truth"][run] = data["truth"]
        arrays["clean_observations"][run] = data["observations"]
        for index, label in enumerate(labels):
            result = run_filter(data, label)
            metrics[label].append({"seed": seed, **final_metrics(result)})
            for record in result["records"]:
                step = record["step"]
                arrays["posterior_mean"][index, run, step] = record["posterior"]["mean"]
                arrays["posterior_covariance"][index, run, step] = record["posterior"][
                    "covariance"
                ]
                if record["predicted"]:
                    arrays["predicted_mean"][index, run, step] = record["predicted"][
                        "mean"
                    ]
                    arrays["predicted_covariance"][index, run, step] = record[
                        "predicted"
                    ]["covariance"]
                if record["observation"] is not None:
                    arrays["used_observation"][index, run, step] = record["observation"]
                    arrays["observed"][index, run, step] = True
                    arrays["accepted"][index, run, step] = record["innovation"][
                        "accepted"
                    ]
                    arrays["innovation_residual"][index, run, step] = record[
                        "innovation"
                    ]["residual"]
                    arrays["innovation_covariance"][index, run, step] = record[
                        "innovation"
                    ]["covariance"]
                    arrays["nis"][index, run, step] = record["innovation"]["nis"]
                arrays["injected_outlier"][index, run, step] = record[
                    "injected_outlier"
                ]
        if (run + 1) % 40 == 0:
            print(f"Completed {run + 1}/{runs} independent trajectories", flush=True)
    np.savez_compressed(out / "results/monte-carlo.npz", **arrays)
    write_json(out / "results/final-metrics.json", metrics)
    write_json(
        out / "results/summary.json",
        [summarize(label, rows) for label, rows in metrics.items()],
    )
    write_json(
        out / "results/archive-schema.json",
        {
            "configuration_axis": labels,
            "run_axis": "seeds array; independent runs, paired across configurations",
            "time_axis": "0 prior; 1..60 after prediction/update; dt=0.5 seconds",
            "state_order": ["x", "y", "vx", "vy"],
            "missing": "NaN for unavailable observations/innovations and predicted prior at index 0; use observed mask",
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
        },
    )
    illustrative = simulate(20261121)
    write_json(out / "results/illustrative-input.json", illustrative)
    for label in labels:
        write_json(
            out / "results/illustrative" / f"{label}.json",
            run_filter(illustrative, label),
        )
    for label in labels:
        for _ in range(2):
            run_filter(illustrative, label)
    order = [label for _ in range(samples) for label in labels]
    random.Random(14992).shuffle(order)
    timing = []
    for ordinal, label in enumerate(order):
        start = perf_counter_ns()
        run_filter(illustrative, label)
        timing.append(
            {
                "ordinal": ordinal + 1,
                "configuration": label,
                "elapsed_ns": perf_counter_ns() - start,
                "steps": STEPS,
            }
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(timing[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(timing)
    write(out / "results/timings.csv", stream.getvalue())
    write_json(
        out / "results/environment.json",
        {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "monte_carlo_runs": runs,
            "monte_carlo_seed": 661024,
            "illustrative_seed": 20261121,
            "timing": {
                "samples_per_configuration": samples,
                "warmups": 2,
                "order_seed": 14992,
                "scope": "60 filtering steps including array validation and Python result construction; excludes simulation, JSON encoding, disk writes and plotting",
            },
        },
    )
    from scripts.report import generate

    generate(out)
    if out != ROOT:
        for name in ("equations.md", "methodology.md", "archive.md"):
            write(
                out / "docs" / name, (ROOT / "docs" / name).read_text(encoding="utf-8")
            )
    paths = [
        *out.glob("results/**/*"),
        *out.glob("docs/figures/*"),
        out / "docs/report.html",
    ]
    write_json(
        out / "results/provenance.json",
        {
            "source_sha256_lf": sources(),
            "artifact_sha256": {
                file.relative_to(out)
                .as_posix(): hashlib.sha256(file.read_bytes())
                .hexdigest()
                for file in sorted(paths)
                if file.is_file() and file.name != "provenance.json"
            },
        },
    )
    from scripts.verify_results import verify_artifacts

    verify_artifacts(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if not 10 <= args.runs <= 250 or not 1 <= args.samples <= 100:
        parser.error("runs must be 10..250 and samples 1..100")
    study(args.output_dir.resolve(), args.runs, args.samples)
