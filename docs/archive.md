# Archive, provenance and reproduction

`results/monte-carlo.npz` is a compressed NumPy archive containing no pickled objects. Load with `numpy.load(path, allow_pickle=False)`. The [machine-readable schema](../results/archive-schema.json) lists every array's shape and dtype.

Configuration order is matched, low_process_noise, sparse_measurements, outliers_ungated, outliers_gated. Posterior arrays have axes configuration × independent run × time × state (× state for covariance). Time 0 is the prior; times 1–60 follow prediction and optional correction. State order is x, y, vx, vy. All covariances, including off-diagonal terms, are retained.

Stored arrays include:

- Independent run seeds, latent truth and clean observations.
- Predicted and posterior means/covariances.
- Actually used observations, innovations, innovation covariances and NIS.
- Observation-present, accepted and injected-fault boolean masks.

Missing observations/innovations and predictions at prior index 0 are represented by NaN, disambiguated by the masks and schema. Posterior means/covariances remain finite at every time. Saved JSON illustrative runs use null for missing records and contain no nonstandard numeric constants.

The illustrative JSON records feed the browser's canvas replay. It draws saved values; it does not rerun the filter or infer a new trajectory. Playback is paused initially and begins only when requested. The slider supports keyboard input.

```bash
python scripts/verify_results.py
python scripts/run_study.py --output-dir build/reproduction
python scripts/verify_results.py --output-dir build/reproduction
```

The verifier checks source/artifact digests, all saved state/covariance transitions, observation schedules/faults/gate decisions, covariance symmetry/PSD tolerance, every final-time metric and confidence summary. It replays the illustrative trajectory with the filter and checks actual timing-row structure. Numerical agreement uses tolerances appropriate to floating-point calculations; timestamps, archive container metadata and timings differ on a fresh run.

Source hashes normalize CRLF to LF. JSON, CSV, HTML, SVG and logs are written with LF before hashing. NPZ/PNG/video are marked binary in Git. Optional media capture helpers and `tests/test_media.py` are outside solve-source provenance. Captures can therefore be attached without replacing original measurements.

The source hash set covers filtering/experiment code, mathematical tests, study orchestration, verification and report generation. It detects accidental inconsistency; it does not authenticate authorship or establish a trusted external timestamp.

CI uses Python 3.13 and pinned dependencies, verifies the preserved study, then creates a fresh 20-run study under ignored `build/ci`. It never rewrites the committed experiment in the normal CI path.
