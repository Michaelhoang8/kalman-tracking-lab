"""Record actual browser playback of saved filter states; optional media tool."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "docs/replay"
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with tempfile.TemporaryDirectory(
        prefix="kalman-replay-"
    ) as temporary, sync_playwright() as pw:
        options = {"headless": True}
        if Path(
            "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        ).exists():
            options["channel"] = "msedge"
        browser = pw.chromium.launch(**options)
        context = browser.new_context(
            viewport={"width": 1200, "height": 640},
            record_video_dir=temporary,
            record_video_size={"width": 1200, "height": 640},
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((ROOT / "docs/report.html").as_uri(), wait_until="networkidle")
        page.locator(".replay").scroll_into_view_if_needed()
        page.locator("#step").fill("0")
        page.locator("#play").click()
        page.wait_for_function(
            "Number(document.getElementById('step').value)>=50", timeout=12000
        )
        page.locator("#play").click()
        step = int(page.locator("#step").input_value())
        page.locator(".replay").screenshot(path=str(output / "replay-frame.png"))
        video = page.video
        context.close()
        raw = Path(video.path())
        browser.close()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output / "canvas-replay.mp4"),
            ],
            check=True,
        )
    assert not errors, errors
    metadata = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Actual automated browser playback of saved illustrative states, steps 0 through recorded last_step.",
        "last_step": step,
        "viewport": [1200, 640],
        "timing": "Illustrative playback around 160 milliseconds per saved step; not filtering execution time.",
        "source_records_sha256": digest(ROOT / "results/illustrative/matched.json"),
        "source_input_sha256": digest(ROOT / "results/illustrative-input.json"),
        "report_sha256_lf": hashlib.sha256(
            (ROOT / "docs/report.html").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "capture_script_sha256_lf": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "video_sha256": digest(output / "canvas-replay.mp4"),
        "frame_sha256": digest(output / "replay-frame.png"),
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"Recorded saved-state replay through step {step}; hashes retained in docs/replay/manifest.json"
    )


if __name__ == "__main__":
    main()
