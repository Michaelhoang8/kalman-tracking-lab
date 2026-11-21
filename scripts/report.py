"""Static scientific figures plus a browser replay of saved filter states."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COLORS = {
    "matched": "#166b8f",
    "low_process_noise": "#c1503e",
    "sparse_measurements": "#8267a4",
    "outliers_ungated": "#ba6a32",
    "outliers_gated": "#237e70",
}
LABELS = {
    "matched": "Matched model",
    "low_process_noise": "Q underestimated ×30",
    "sparse_measurements": "Sparse + missing",
    "outliers_ungated": "Outliers, no gate",
    "outliers_gated": "Outliers, gated",
}
BACKGROUND = "#f4f7f9"
THRESHOLD_95 = -2 * np.log(0.05)


def ellipse(axis, mean, cov, color, alpha=0.25):
    values, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    widths = 2 * np.sqrt(np.maximum(values, 0) * THRESHOLD_95)
    axis.add_patch(
        Ellipse(
            mean,
            width=widths[1],
            height=widths[0],
            angle=angle,
            edgecolor=color,
            facecolor=color,
            lw=0.8,
            alpha=alpha,
        )
    )


def save(figure, path):
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    if path.suffix == ".svg":
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    plt.close(figure)


def generate(out: Path):
    def read(path):
        return json.loads((out / path).read_text(encoding="utf-8"))

    data = read("results/illustrative-input.json")
    summaries = read("results/summary.json")
    environment = read("results/environment.json")
    results = {
        row["configuration"]: read(f'results/illustrative/{row["configuration"]}.json')
        for row in summaries
    }
    arrays = {}
    for label, result in results.items():
        records = result["records"]
        arrays[label] = {
            "mean": np.asarray([row["posterior"]["mean"] for row in records]),
            "cov": np.asarray([row["posterior"]["covariance"] for row in records]),
            "time": np.asarray([row["time"] for row in records]),
        }
    truth = np.asarray(data["truth"])
    observed = np.asarray(data["observations"])
    figures = out / "docs/figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "text.color": "#233742",
            "axes.labelcolor": "#233742",
            "xtick.color": "#60717a",
            "ytick.color": "#60717a",
            "axes.edgecolor": "#bac7cc",
            "grid.color": "#dce4e8",
            "svg.hashsalt": "kalman-tracking-lab",
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 5.4), constrained_layout=True)
    ax.scatter(
        observed[:, 0],
        observed[:, 1],
        s=10,
        color="#a1b2bb",
        label="Noisy position observations",
        zorder=1,
    )
    ax.plot(truth[:, 0], truth[:, 1], color="#233742", lw=1.6, label="Synthetic truth")
    mean, cov = arrays["matched"]["mean"], arrays["matched"]["cov"]
    ax.plot(
        mean[:, 0],
        mean[:, 1],
        color=COLORS["matched"],
        lw=1.7,
        label="Matched filter mean",
    )
    for step in range(5, 61, 10):
        ellipse(ax, mean[step, :2], cov[step, :2, :2], COLORS["matched"])
    ax.set(xlabel="x position (m)", ylabel="y position (m)", aspect="equal")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.5)
    save(fig, figures / "trajectory.svg")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.6), constrained_layout=True)
    for label in ("matched", "low_process_noise"):
        array = arrays[label]
        error = np.linalg.norm(array["mean"][:, :2] - truth[:, :2], axis=1)
        area = np.pi * THRESHOLD_95 * np.sqrt(np.linalg.det(array["cov"][:, :2, :2]))
        axes[0].plot(array["time"], error, color=COLORS[label], label=LABELS[label])
        axes[1].plot(array["time"], area, color=COLORS[label])
    axes[0].set(xlabel="Time (s)", ylabel="Position error norm (m)")
    axes[1].set(xlabel="Time (s)", ylabel="95% marginal ellipse area (m²)")
    axes[0].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.5)
    save(fig, figures / "noise-model.svg")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    sparse = arrays["sparse_measurements"]
    axes[0].plot(truth[:, 0], truth[:, 1], color="#233742", label="Truth")
    axes[0].plot(
        sparse["mean"][:, 0],
        sparse["mean"][:, 1],
        color=COLORS["sparse_measurements"],
        label="Sparse estimate",
    )
    for step in (24, 28, 32, 36, 42):
        ellipse(
            axes[0],
            sparse["mean"][step, :2],
            sparse["cov"][step, :2, :2],
            COLORS["sparse_measurements"],
            0.18,
        )
    axes[0].set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    axes[0].legend(frameon=False, fontsize=9)
    for label in ("matched", "sparse_measurements"):
        array = arrays[label]
        axes[1].plot(
            array["time"],
            np.sqrt(array["cov"][:, 0, 0]),
            color=COLORS[label],
            label=LABELS[label],
        )
    axes[1].axvspan(12.5, 18, color="#d9d2e3", alpha=0.5, label="Steps 25–36 missing")
    axes[1].set(xlabel="Time (s)", ylabel="Posterior x standard deviation (m)")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.5)
    save(fig, figures / "missing.svg")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7), constrained_layout=True)
    for label in ("outliers_ungated", "outliers_gated"):
        array = arrays[label]
        axes[0].plot(
            array["time"],
            np.linalg.norm(array["mean"][:, :2] - truth[:, :2], axis=1),
            color=COLORS[label],
            label=LABELS[label],
        )
        records = results[label]["records"][1:]
        nis = np.asarray([row["innovation"]["nis"] for row in records])
        axes[1].plot(array["time"][1:], nis, color=COLORS[label], alpha=0.85, lw=1.2)
    gated = results["outliers_gated"]["records"][1:]
    rejected = [row for row in gated if not row["innovation"]["accepted"]]
    axes[1].scatter(
        [row["time"] for row in rejected],
        [row["innovation"]["nis"] for row in rejected],
        marker="x",
        color="#233742",
        s=32,
        label="Gate rejection",
    )
    axes[1].axhline(
        -2 * np.log(0.01), ls="--", color="#233742", lw=1, label="Threshold 9.21"
    )
    for ax in axes:
        for time in (9, 18, 24):
            ax.axvline(time, color="#cad4d9", lw=1)
        ax.grid(alpha=0.4)
    axes[0].set(xlabel="Time (s)", ylabel="Position error norm (m)")
    axes[1].set(xlabel="Time (s)", ylabel="Pre-decision NIS", yscale="log")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, figures / "gating.svg")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7), constrained_layout=True)
    for index, row in enumerate(summaries):
        color = COLORS[row["configuration"]]
        coverage, interval = row["position_coverage"], row["coverage_wilson_95"]
        axes[0].errorbar(
            coverage,
            index,
            xerr=[[coverage - interval[0]], [interval[1] - coverage]],
            fmt="o",
            color=color,
            capsize=3,
        )
        axes[1].errorbar(
            row["mean_nees"],
            index,
            xerr=row["nees_mc_standard_error"],
            fmt="o",
            color=color,
            capsize=3,
        )
    labels = [LABELS[row["configuration"]] for row in summaries]
    axes[0].set(
        yticks=range(5),
        yticklabels=labels,
        xlabel="Final-time position coverage / Wilson 95% interval",
        xlim=(0, 1.02),
    )
    axes[1].set(
        yticks=range(5),
        yticklabels=[],
        xlabel="Mean final NEES ± 1 Monte Carlo standard error",
    )
    axes[0].axvline(0.95, color="#8799a3", ls="--", lw=1)
    axes[1].axvline(4, color="#8799a3", ls="--", lw=1)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.4)
    save(fig, figures / "consistency.svg")

    with (out / "results/timings.csv").open(newline="", encoding="utf-8") as handle:
        timing = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(11.8, 2.7), constrained_layout=True)
    for index, label in enumerate(results):
        values = [
            int(row["elapsed_ns"]) / 1e6
            for row in timing
            if row["configuration"] == label
        ]
        ax.scatter(
            values,
            [index + ((j % 5) - 2) * 0.035 for j in range(len(values))],
            s=13,
            color=COLORS[label],
            alpha=0.6,
        )
        ax.plot(
            [np.median(values)] * 2, [index - 0.16, index + 0.16], color="#233742", lw=2
        )
    ax.set(
        yticks=range(5),
        yticklabels=[LABELS[label] for label in results],
        xlabel="Elapsed milliseconds per 60-step filtering run",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.4)
    save(fig, figures / "timings.svg")

    table = ""
    for row in summaries:
        interval = row["coverage_wilson_95"]
        table += f'<tr><th>{LABELS[row["configuration"]]}</th><td>{row["position_rmse"]:.2f}</td><td>{row["velocity_rmse"]:.2f}</td><td>{row["position_coverage"]:.1%} <small>[{interval[0]:.1%}, {interval[1]:.1%}]</small></td><td>{row["mean_nees"]:.2f} ± {row["nees_mc_standard_error"]:.2f}</td></tr>'
    html = TEMPLATE.replace("{{TABLE}}", table).replace(
        "{{RUNS}}", str(environment["monte_carlo_runs"])
    )
    html = html.replace(
        "{{REPLAY}}",
        json.dumps(
            results["matched"]["records"], separators=(",", ":"), allow_nan=False
        ),
    )
    (out / "docs/report.html").write_text(html, encoding="utf-8", newline="\n")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kalman tracking · estimation study</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:#233742;font:16px/1.55 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:0 38px}nav{display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #bfccd3;padding:22px 0;font:12px ui-monospace,monospace}a{color:#166b8f}nav a{text-decoration:none}section{padding:40px 0 48px;border-bottom:1px solid #bfccd3}h1{font-size:clamp(34px,4.7vw,58px);font-weight:500;line-height:1.1;letter-spacing:-1.8px;margin:28px 0 18px}h2{font-size:29px;font-weight:500;line-height:1.2;margin:0 0 22px;letter-spacing:-.5px}h3{font-size:17px;margin:0 0 10px}.kicker{font:12px ui-monospace,monospace;letter-spacing:1px;color:#657b87;margin-top:28px}.lead{font-size:18px;max-width:880px}p{max-width:930px}figure{margin:24px 0}figure img{width:100%;height:auto;display:block}figcaption{color:#657b87;font-size:13px;margin-top:10px}.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:40px}.two>*{min-width:0}.equation{font:14px/1.9 ui-monospace,monospace;background:#e8eef2;padding:22px;white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}.note{border-left:2px solid #8267a4;padding-left:18px}.scroll{max-width:100%;overflow-x:auto}.scroll:focus-visible{outline:2px solid #166b8f;outline-offset:4px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border-bottom:1px solid #d1dce2;text-align:left}th{font-weight:500}thead th{font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:#657b87}td small{display:block;color:#657b87}.links{display:flex;gap:22px;flex-wrap:wrap;margin-top:22px;font-size:13px}.replay{border:1px solid #c6d4dc;padding:16px;margin-top:22px}canvas{display:block;width:100%;height:auto;background:#f4f7f9}.controls{display:flex;align-items:center;gap:15px;flex-wrap:wrap;margin-top:10px}.controls input{flex:1;min-width:160px}button{font:inherit;color:#233742;border:1px solid #9bafb9;border-radius:3px;padding:7px 18px;background:#f4f7f9;cursor:pointer}button:focus-visible,input:focus-visible{outline:2px solid #166b8f}footer{font-size:12px;color:#657b87;padding:25px 0 40px}@media(max-width:700px){main{padding:0 18px}h1{letter-spacing:-1px}h2{font-size:24px}.two{grid-template-columns:minmax(0,1fr);gap:18px}.scroll table{min-width:650px}section{padding:30px 0}nav{gap:14px}}
</style></head><body><main>
<section id="overview"><nav><a href="#overview">Trajectory</a><a href="#model">Noise model</a><a href="#missing">Missing observations</a><a href="#gating">Gating</a><a href="#consistency">Consistency</a><a href="#reproduction">Replay / evidence</a></nav><div class="kicker">STATE ESTIMATION / SYNTHETIC TRACKING</div><h1>Position, velocity<br>and estimated uncertainty</h1><p class="lead">A linear Kalman filter estimates a two-dimensional trajectory from noisy positions. These plots show saved states from the implemented filter, with comparisons against independent Gaussian calculations.</p><figure><img src="figures/trajectory.svg" alt="Synthetic truth, noisy positions, estimated path and marginal uncertainty ellipses"><figcaption>Illustrative seed 20261121, declared before comparison. Ellipses are 95% marginal position contours at selected times, not a simultaneous confidence region for the entire path.</figcaption></figure></section>
<section id="model"><h2>Process noise and covariance updates</h2><div class="two"><div><p>The state is <code>[x, y, vx, vy]</code>. Continuous white acceleration with spectral density q = 0.18 m²/s³ is integrated over Δt = 0.5 seconds. The filter uses linear solves and the Joseph covariance update.</p><pre class="equation">F = [ I   Δt I ]
    [ 0      I ]

Q = q [ (Δt³/3) I  (Δt²/2) I ]
      [ (Δt²/2) I       Δt I ]</pre></div><div><p>The underestimated-noise filter uses q / 30 on the same observations. Its uncertainty can shrink while its error grows. The comparison below is descriptive for one recorded trajectory.</p><pre class="equation">S = H P⁻ Hᵀ + R
solve S Kᵀ = H P⁻

P⁺ = (I − K H) P⁻ (I − K H)ᵀ
     + K R Kᵀ</pre></div></div><figure><img src="figures/noise-model.svg" alt="Position error and marginal uncertainty area under matched and underestimated process noise"></figure></section>
<section id="missing"><h2>Prediction during missing observations</h2><p>The sparse case receives every third measurement, with steps 25–36 omitted entirely. Missing observations cause prediction only. The position covariance expands during the gap and contracts when updates resume.</p><figure><img src="figures/missing.svg" alt="Sparse observation trajectory and position standard deviation during an observation gap"><figcaption>The highlighted interval is the declared missing-observation window. Every covariance comes from the saved posterior array; no uncertainty is drawn by hand.</figcaption></figure></section>
<section id="gating"><h2>Innovation gating under injected outliers</h2><p>Both variants receive the same additive position faults at steps 18, 36 and 48. A candidate observation is accepted when its normalized innovation squared is at most 9.21, the nominal 99% χ²₂ threshold.</p><figure><img src="figures/gating.svg" alt="Position errors with and without gating and pre-decision innovation statistics"><figcaption>Vertical lines identify injected faults; crosses mark actual gate rejections. The gate can also reject legitimate motion and does not identify the cause of a large innovation.</figcaption></figure><p class="note">Gating selects observations using their residuals and changes subsequent conditioning. Accepted-only NIS is selection-biased. Gated covariance and consistency statistics here are descriptive, without a claim of nominal Gaussian calibration.</p></section>
<section id="consistency"><h2>Fixed-time Monte Carlo assessment</h2><p>{{RUNS}} independently seeded trajectories are evaluated at step 60 only. Configurations share each trajectory for paired comparisons. Repeated time steps are not counted as independent coverage trials.</p><div class="scroll" tabindex="0" role="region" aria-label="Scrollable final-time Monte Carlo results"><table><thead><tr><th>Configuration</th><th>Position RMSE (m)</th><th>Velocity RMSE (m/s)</th><th>95% position coverage / Wilson interval</th><th>Mean NEES ± MC SE</th></tr></thead><tbody>{{TABLE}}</tbody></table></div><figure><img src="figures/consistency.svg" alt="Final-time position coverage and mean normalized state error across independent runs"><figcaption>Dashed references: nominal 95% position coverage and mean NEES = 4 for a correctly specified ungated four-state Gaussian model. Wilson intervals use independent final-time indicators. NEES bars are one estimated Monte Carlo standard error, not 95% intervals.</figcaption></figure></section>
<section id="reproduction"><h2>Saved-state replay and verification</h2><p>The replay reads the saved illustrative records. It performs no filtering in the browser. Scrub through the observed position, posterior mean and current 95% position ellipse.</p><div class="replay"><canvas id="replay" width="1100" height="340" role="img" aria-label="Saved trajectory replay showing truth, filter estimate and uncertainty ellipse"></canvas><div class="controls"><button id="play" type="button">Play</button><label for="step">Time step</label><input id="step" type="range" min="0" max="60" value="60"><output id="time" for="step" aria-live="polite">30.0 s</output></div></div><figure><img src="figures/timings.svg" alt="Actual local durations of 60-step filtering runs"><figcaption>Actual repeated runs after two warmups, in a seeded interleaved order. Timings include validation and Python result construction; simulation, disk I/O and plotting are excluded.</figcaption></figure><p>Analytic scalar checks, 24 independent batch-Gaussian least-squares references, process-noise composition, gate boundaries, covariance checks and complete archive replay validate the implementation. These are synthetic linear-model experiments, not a deployed robotics system.</p><div class="links"><a href="../results/summary.json">Summary JSON</a><a href="../results/monte-carlo.npz">Complete state archive</a><a href="../results/provenance.json">Provenance</a><a href="equations.md">Equations</a><a href="methodology.md">Methodology</a></div></section>
<footer>AI-assisted implementation and documentation · Synthetic trajectories · NumPy linear algebra; no prebuilt filtering package.</footer>
</main><script>
const records={{REPLAY}},canvas=document.getElementById('replay'),ctx=canvas.getContext('2d'),slider=document.getElementById('step'),output=document.getElementById('time'),button=document.getElementById('play');
const all=records.flatMap(r=>[r.truth.slice(0,2),r.posterior.mean.slice(0,2),...(r.observation?[r.observation]:[])]),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
const loX=Math.min(...xs)-4,hiX=Math.max(...xs)+4,loY=Math.min(...ys)-4,hiY=Math.max(...ys)+4,scale=Math.min(1000/(hiX-loX),260/(hiY-loY));
const px=x=>550+(x-(loX+hiX)/2)*scale,py=y=>170-(y-(loY+hiY)/2)*scale;
function draw(){const step=Number(slider.value),r=records[step];ctx.clearRect(0,0,1100,340);ctx.font='14px sans-serif';ctx.fillStyle='#233742';ctx.fillText('Truth',22,24);ctx.fillStyle='#166b8f';ctx.fillText('Posterior estimate',92,24);ctx.fillStyle='#97aeb9';ctx.fillText('Observation',250,24);
for(const [key,color] of [['truth','#233742'],['posterior','#166b8f']]){ctx.beginPath();records.slice(0,step+1).forEach((p,i)=>{const v=key==='truth'?p.truth:p.posterior.mean;i?ctx.lineTo(px(v[0]),py(v[1])):ctx.moveTo(px(v[0]),py(v[1]));});ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
if(r.observation){ctx.fillStyle='#97aeb9';ctx.beginPath();ctx.arc(px(r.observation[0]),py(r.observation[1]),4,0,Math.PI*2);ctx.fill();}
const p=r.posterior.covariance,a=p[0][0],b=p[0][1],d=p[1][1],term=Math.sqrt((a-d)**2+4*b*b),large=(a+d+term)/2,small=(a+d-term)/2,angle=-.5*Math.atan2(2*b,a-d),m=r.posterior.mean;
ctx.beginPath();ctx.ellipse(px(m[0]),py(m[1]),Math.sqrt(Math.max(0,large)*5.991464547)*scale,Math.sqrt(Math.max(0,small)*5.991464547)*scale,angle,0,Math.PI*2);ctx.fillStyle='#166b8f20';ctx.fill();ctx.strokeStyle='#166b8f';ctx.lineWidth=1;ctx.stroke();output.textContent=r.time.toFixed(1)+' s / step '+step;}
let playing=false,last=0;button.onclick=()=>{playing=!playing;button.textContent=playing?'Pause':'Play';if(playing&&Number(slider.value)===60)slider.value=0;draw();};slider.oninput=()=>{playing=false;button.textContent='Play';draw();};
function animate(now){if(playing&&now-last>160){slider.value=Number(slider.value)+1;last=now;if(Number(slider.value)>=60){playing=false;button.textContent='Play';}draw();}requestAnimationFrame(animate);}draw();requestAnimationFrame(animate);
</script></body></html>"""
