"""Generate and filter one reproducible synthetic trajectory."""

import argparse
import json
from pathlib import Path
import sys
from .experiment import CONFIGURATIONS, simulate, run_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="matched")
    parser.add_argument("--output", type=Path, default=Path("build/trajectory.json"))
    args = parser.parse_args()
    try:
        data = simulate(args.seed, args.steps)
        result = run_filter(data, args.configuration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"input": data, "result": result}, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"Saved {len(result['records'])} states to {args.output}")
        return 0
    except (ValueError, OSError) as error:
        print(f"tracking: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
