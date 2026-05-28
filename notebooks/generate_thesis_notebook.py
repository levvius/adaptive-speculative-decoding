"""Generate notebooks/thesis_plots.ipynb with 5 mandatory thesis-defense plots.

Run:  .venv/bin/python notebooks/generate_thesis_notebook.py

The notebook itself contains all logic to load run artifacts, compute paired
statistics, and render plots saved as PDFs under reports/thesis_figs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def md_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


CELLS: list[nbf.NotebookNode] = []


CELLS.append(
    md_cell(
        """# JointAdaSpec — Thesis-Defense Plots

Five mandatory plots for the defense slide deck plus a κ-sweep bonus and a paired-EM significance table.

Generated from the run artifacts under `outputs/` and the condition checks under `reports/conditions_*.json`. All plots are saved as PDF under `reports/thesis_figs/` and rendered inline.

**Plots**
1. Headline Pareto (EM vs tok/s, both model pairs)
2. Paired EM significance bars with 95% CIs and p-values
3. Threshold surface T*(H, K) for k ∈ {0, 2, 4}
4. C1–C4 condition diagnostics + Theorem-C empirical bound
5. Speedup vs vanilla speculative

**Bonus**
- κ-sweep on existing traces → empirical Pareto front (Theorem 2.4 complement)

See `reports/theory_improvements_2026-05-15.md` for the theoretical framing.
"""
    )
)


CELLS.append(
    code_cell(
        '''import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json

REPO_ROOT = Path.cwd()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

FIGS_DIR = REPO_ROOT / "reports" / "thesis_figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Slide-deck friendly defaults
sns.set_theme(style="whitegrid", context="talk")
mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["pdf.fonttype"] = 42  # editable text in PDFs

# Locked anchor runs (final sprint, 500 prompts x 3 seeds = paired n=1500).
# Updated 2026-05-20 from the small-n crosscheck/k8 dirs to the lock dirs so the
# thesis figures reflect the locked numbers (14B: +4.07% p=0.02; 7B: null).
RUN_14B = {
    "label": "Qwen 14B → 0.5B",
    "benchmark_csv": REPO_ROOT / "outputs" / "jointadaspec_qwen14b_0p5b_lock_2026-05-14" / "03_bench_gsm8k" / "benchmark.csv",
    "policy_npz": REPO_ROOT / "outputs" / "jointadaspec_qwen14b_0p5b_2026-04-28" / "02_solve" / "policy.npz",
    "conditions_json": REPO_ROOT / "reports" / "conditions_qwen14b_0p5b_2026-04-28.json",
}
RUN_7B = {
    "label": "Qwen 7B → 1.5B",
    "benchmark_csv": REPO_ROOT / "outputs" / "jointadaspec_qwen7b_1p5b_quality_lock_2026-05-14" / "03_bench_gsm8k" / "benchmark.csv",
    "policy_npz": REPO_ROOT / "outputs" / "jointadaspec_qwen7b_1p5b_quality_2026-05-05" / "02_solve" / "policy.npz",
    "conditions_json": REPO_ROOT / "reports" / "conditions_qwen7b_1p5b_quality_2026-05-05.json",
}

METHOD_RENAME = {
    "vanilla_ar": "target_only",
    "target_only": "target_only",
    "fixed_sd": "speculative",
    "speculative": "speculative",
    "cascade_length_then_verif": "cascade_len_then_verif",
    "cascade_verif_then_length": "cascade_verif_then_length",
    "jointadaspec": "jointadaspec",
}
METHOD_ORDER = ["target_only", "speculative", "cascade_verif_then_length", "jointadaspec"]
METHOD_COLORS = {
    "target_only": "#666666",
    "speculative": "#4C72B0",
    "cascade_verif_then_length": "#DD8452",
    "cascade_len_then_verif": "#937860",
    "jointadaspec": "#55A868",
}
print("Repo root:", REPO_ROOT)
print("Figs dir:", FIGS_DIR)
print("Available 14B benchmark:", RUN_14B["benchmark_csv"].exists())
print("Available 7B benchmark:", RUN_7B["benchmark_csv"].exists())
'''
    )
)


CELLS.append(
    md_cell("## Data loading helpers\n\nLoad per-prompt benchmark records, rename methods to the paper-style canonical names, and compute paired EM statistics (joint vs baseline). Bootstrap 95% CIs use 10000 resamples for stable tail estimates.")
)


CELLS.append(
    code_cell(
        '''def load_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = df["decoder"].map(METHOD_RENAME).fillna(df["decoder"])
    return df


def aggregate_methods(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("method").agg(
        n=("prompt_idx", "count"),
        em=("gsm8k_exact_match", "mean"),
        tps=("tokens_per_sec", "mean"),
        acceptance=("acceptance_rate", "mean"),
    ).reset_index()
    return grouped


def paired_em_diff(df: pd.DataFrame, method: str, baseline: str = "target_only", n_boot: int = 10000, seed: int = 17):
    """Paired EM diff and 95% bootstrap CI computed at the prompt-id level."""
    join = df.pivot_table(index="prompt_idx", columns="method", values="gsm8k_exact_match", aggfunc="mean")
    if method not in join.columns or baseline not in join.columns:
        return None
    pair = join[[method, baseline]].dropna()
    diffs = (pair[method] - pair[baseline]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    n = diffs.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = diffs[idx].mean()
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    # Sign-flip permutation test
    perm_samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        signs = rng.choice([-1, 1], size=n)
        perm_samples[i] = (signs * diffs).mean()
    observed = diffs.mean()
    p = float(np.mean(np.abs(perm_samples) >= abs(observed)))
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    return {
        "n": n,
        "diff": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def speed_summary(df: pd.DataFrame) -> dict:
    """Method → mean tok/s, plus speedup ratios vs speculative and baseline."""
    methods = aggregate_methods(df)
    by_method = dict(zip(methods["method"], methods["tps"]))
    spec_tps = by_method.get("speculative", 1.0) or 1.0
    base_tps = by_method.get("target_only", 1.0) or 1.0
    out = {}
    for method, tps in by_method.items():
        out[method] = {
            "tps": float(tps),
            "vs_speculative": float(tps / spec_tps),
            "vs_target_only": float(tps / base_tps),
        }
    return out


df_14b = load_benchmark(RUN_14B["benchmark_csv"])
df_7b = load_benchmark(RUN_7B["benchmark_csv"])

agg_14b = aggregate_methods(df_14b)
agg_7b = aggregate_methods(df_7b)
print("=== 14B/0.5B ===")
print(agg_14b.to_string(index=False))
print()
print("=== 7B/1.5B ===")
print(agg_7b.to_string(index=False))
'''
    )
)


CELLS.append(md_cell("## Plot 1 — Headline Pareto: EM vs throughput\n\nThe single most important slide. Y-axis: GSM8K exact-match. X-axis: tokens per second. Each method is a point per model pair. JointAdaSpec sits on (or above) the Pareto front against vanilla speculative."))


CELLS.append(
    code_cell(
        '''fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, (df, info) in zip(axes, [(df_14b, RUN_14B), (df_7b, RUN_7B)]):
    agg = aggregate_methods(df)
    for _, row in agg.iterrows():
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        ax.scatter(
            row["tps"], row["em"],
            s=260,
            color=METHOD_COLORS.get(method, "#444"),
            edgecolor="black", linewidth=1.2,
            label=method,
            zorder=3,
        )
        ax.annotate(
            method,
            (row["tps"], row["em"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=11,
        )
    ax.set_title(info["label"])
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("GSM8K exact match")
    ax.grid(True, alpha=0.3)

fig.suptitle("Pareto front — quality vs throughput", y=1.02, fontsize=16)
fig.tight_layout()
out_path = FIGS_DIR / "fig1_pareto.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Plot 2 — Paired EM significance bars\n\nFor each method `m`, plot `EM(m) − EM(target_only)` per-prompt-paired, with 95% bootstrap CI and a permutation-test p-value. JointAdaSpec is the rightmost bar in each pair; its CI ought to clear zero (significance) or come close. After Run 1 lock, the 14B bar should narrow noticeably."))


CELLS.append(
    code_cell(
        '''rows = []
for df, info in [(df_14b, RUN_14B), (df_7b, RUN_7B)]:
    for method in METHOD_ORDER:
        if method == "target_only":
            continue
        stats = paired_em_diff(df, method=method)
        if stats is None:
            continue
        rows.append({
            "pair": info["label"],
            "method": method,
            **stats,
        })
diff_df = pd.DataFrame(rows)
print(diff_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (df, info) in zip(axes, [(df_14b, RUN_14B), (df_7b, RUN_7B)]):
    pair_label = info["label"]
    sub = diff_df[diff_df["pair"] == pair_label]
    xs = np.arange(len(sub))
    ax.bar(
        xs, sub["diff"] * 100.0,
        yerr=[(sub["diff"] - sub["ci_low"]) * 100.0, (sub["ci_high"] - sub["diff"]) * 100.0],
        capsize=8,
        color=[METHOD_COLORS.get(m, "#888") for m in sub["method"]],
        edgecolor="black", linewidth=1.0,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["method"], rotation=15, ha="right")
    ax.set_title(pair_label)
    ax.set_ylabel("Δ EM vs target_only (%)")
    ax.grid(True, axis="y", alpha=0.3)
    for x, (_, row) in zip(xs, sub.iterrows()):
        label = f"p={row['p_value']:.3f}"
        ax.annotate(label, (x, row["ci_high"] * 100.0 + 0.5), ha="center", fontsize=10)

fig.suptitle("Paired EM advantage with 95% bootstrap CI and permutation p-value", y=1.02, fontsize=15)
fig.tight_layout()
out_path = FIGS_DIR / "fig2_paired_em.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Plot 3 — Threshold surface T*(H, K) for k ∈ {0, 2, 4}\n\nVisualises *what the joint policy learned*. For each `k` slice, the heat-map shows the chosen verification threshold `T` over (H, K). The 14B/0.5B policy is the anchor."))


CELLS.append(
    code_cell(
        '''from jointadaspec.inference.policy import JointAdaSpecPolicy
from jointadaspec.mdp.spaces import ActionSpace, StateSpace

policy = JointAdaSpecPolicy.load(RUN_14B["policy_npz"])
config = policy.config
state_space = StateSpace(config)
action_space = ActionSpace(config)

ks_to_plot = [0, min(2, config.gamma_max), min(4, config.gamma_max)]

fig, axes = plt.subplots(1, len(ks_to_plot), figsize=(5 * len(ks_to_plot), 4.6))

T_values = np.array([a.threshold for a in action_space.actions])

for ax, k in zip(axes, ks_to_plot):
    grid = np.full((config.N_K, config.N_H), np.nan)
    for i_H in range(config.N_H):
        for i_K in range(config.N_K):
            H_center = (i_H + 0.5) * config.H_max / config.N_H
            K_center = (i_K + 0.5) * config.K_max / config.N_K
            s = state_space.encode(H=H_center, K=K_center, k=k)
            a = int(policy.pi_star[s])
            grid[i_K, i_H] = T_values[a]
    im = ax.imshow(
        grid, origin="lower", aspect="auto", cmap="viridis",
        extent=[0, config.H_max, 0, config.K_max],
        vmin=min(config.T_levels), vmax=max(config.T_levels),
    )
    ax.set_title(f"k = {k}")
    ax.set_xlabel("H (entropy)")
    ax.set_ylabel("K (KL divergence)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("threshold T*")

fig.suptitle(f"Joint policy — verification threshold T*(H, K, k) on {RUN_14B['label']}", y=1.02, fontsize=14)
fig.tight_layout()
out_path = FIGS_DIR / "fig3_threshold_surface.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Plot 4 — Conditions C1–C4 + Theorem-C bound\n\nLeft: pass rate of each empirical condition. Right: Theorem-C bound on cascade suboptimality, computed from the C4-violating stationary mass μ*ᴊ(B). See `reports/theory_improvements_2026-05-15.md` for the bound derivation."))


CELLS.append(
    code_cell(
        '''def load_conditions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

cond_14b = load_conditions(RUN_14B["conditions_json"])
cond_7b = load_conditions(RUN_7B["conditions_json"])

def extract_pass_rates(cond: dict) -> dict:
    out = {}
    for key in ("c1", "c2", "c3", "c4", "n1", "n2"):
        info = cond["checks"].get(key, {})
        if "fraction_nonnegative" in info:
            out[key] = float(info["fraction_nonnegative"])
        elif info.get("passed") is True:
            out[key] = 1.0
        elif info.get("passed") is False:
            out[key] = 0.0
        else:
            out[key] = np.nan
    return out

rates_14b = extract_pass_rates(cond_14b)
rates_7b = extract_pass_rates(cond_7b)
print("14B/0.5B:", rates_14b)
print("7B/1.5B :", rates_7b)

# Theorem-C bound on cascade suboptimality.
# Loose upper bound uses union stationary mass from N1.
def theorem_c_bound(cond: dict, gamma: float = 0.99, R_max: float = 2.0) -> dict:
    n1 = cond["checks"].get("n1", {})
    union_mass = float(n1.get("union", {}).get("stationary_mass", np.nan))
    c4 = cond["checks"].get("c4", {})
    c4_pass = float(c4.get("fraction_nonnegative", np.nan))
    # μ*J(B) ≤ min(union_mass, 1 - c4_pass): mass on which joint != cascade
    # AND where C4 fails. Both upper bounds are valid; take the tighter.
    mu_B = min(union_mass, 1.0 - c4_pass)
    bound = 2.0 * R_max * mu_B / (1.0 - gamma)
    return {"mu_B": mu_B, "bound": bound, "union_mass": union_mass, "c4_pass": c4_pass}

bound_14b = theorem_c_bound(cond_14b)
bound_7b = theorem_c_bound(cond_7b)
print()
print("14B/0.5B Theorem-C bound:", bound_14b)
print("7B/1.5B  Theorem-C bound:", bound_7b)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
keys = ["c1", "c2", "c3", "c4", "n1", "n2"]
xs = np.arange(len(keys))
w = 0.4
ax.bar(xs - w/2, [rates_14b[k] for k in keys], width=w, label="14B/0.5B", color="#4C72B0", edgecolor="black")
ax.bar(xs + w/2, [rates_7b[k]  for k in keys], width=w, label="7B/1.5B",  color="#DD8452", edgecolor="black")
ax.axhline(0.9, color="red", linestyle="--", alpha=0.6, label="C3/C4 target 0.9")
ax.set_xticks(xs)
ax.set_xticklabels([k.upper() for k in keys])
ax.set_ylabel("Pass rate / mass")
ax.set_title("Empirical conditions C1–C4, N1–N2")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
pairs = ["14B/0.5B", "7B/1.5B"]
mu_vals = [bound_14b["mu_B"], bound_7b["mu_B"]]
ax.bar(pairs, mu_vals, color=["#4C72B0", "#DD8452"], edgecolor="black")
ax.set_ylabel("μ*J(B): C4-violating occupancy mass")
ax.set_title("Theorem C — cascade suboptimality\\nbounded linearly in μ*J(B)")
for i, v in enumerate(mu_vals):
    ax.annotate(f"≤ {v:.3f}", (i, v + 0.005), ha="center", fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, max(mu_vals) * 1.25 + 0.02)

fig.tight_layout()
out_path = FIGS_DIR / "fig4_conditions.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Plot 5 — Speedup vs vanilla speculative\n\nHeadline number for the slide deck: JointAdaSpec is **2.27× faster than vanilla speculative on 14B/0.5B** at higher quality. On 7B/1.5B the ratio is ~1.56×."))


CELLS.append(
    code_cell(
        '''speed_14b = speed_summary(df_14b)
speed_7b = speed_summary(df_7b)
print("14B:", json.dumps(speed_14b, indent=2))
print("7B :", json.dumps(speed_7b,  indent=2))

fig, ax = plt.subplots(figsize=(9, 5))
methods = ["target_only", "cascade_verif_then_length", "jointadaspec"]
x = np.arange(len(methods))
w = 0.38
vals_14b = [speed_14b[m]["vs_speculative"] for m in methods]
vals_7b  = [speed_7b[m]["vs_speculative"]  for m in methods]
ax.bar(x - w/2, vals_14b, width=w, label="14B → 0.5B", color="#4C72B0", edgecolor="black")
ax.bar(x + w/2, vals_7b,  width=w, label="7B → 1.5B",  color="#DD8452", edgecolor="black")
ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="vanilla speculative (1×)")
for xi, v in zip(x - w/2, vals_14b):
    ax.annotate(f"{v:.2f}×", (xi, v + 0.05), ha="center", fontsize=11)
for xi, v in zip(x + w/2, vals_7b):
    ax.annotate(f"{v:.2f}×", (xi, v + 0.05), ha="center", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.set_ylabel("Throughput ratio vs vanilla speculative")
ax.set_title("Speedup over vanilla speculative decoding")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out_path = FIGS_DIR / "fig5_speedup.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Bonus — κ-sweep Pareto front (Theorem 2.4 complement)\n\nThis cell is a placeholder that runs only if the κ-sweep artifacts are present. Solve the MDP at κ ∈ {0.5, 1.0, 2.0, 5.0} and benchmark each on a small validation slice; then plot in (EM, tok/s) space and verify convexity. Generates `outputs/jointadaspec_qwen7b_1p5b_kappa_sweep/` artifacts when run end-to-end."))


CELLS.append(
    code_cell(
        '''sweep_root = REPO_ROOT / "outputs" / "jointadaspec_qwen7b_1p5b_kappa_sweep"
if not sweep_root.exists():
    print(f"[skip] {sweep_root} not present. Run the κ-sweep first (see thesis_final_summary).")
else:
    rows = []
    for sub in sorted(sweep_root.glob("kappa_*/03_bench_gsm8k/benchmark.csv")):
        kappa = float(sub.parent.parent.name.split("_")[1])
        df = load_benchmark(sub)
        for m in ("jointadaspec", "cascade_verif_then_length"):
            g = df[df["method"] == m]
            if g.empty:
                continue
            rows.append({
                "kappa": kappa,
                "method": m,
                "em": g["gsm8k_exact_match"].mean(),
                "tps": g["tokens_per_sec"].mean(),
            })
    if not rows:
        print("[skip] κ-sweep results empty.")
    else:
        sweep_df = pd.DataFrame(rows)
        print(sweep_df.to_string(index=False))
        fig, ax = plt.subplots(figsize=(8, 5))
        for m, color, label in (("jointadaspec", "#55A868", "joint"),
                                ("cascade_verif_then_length", "#C44E52", "cascade")):
            sub = sweep_df[sweep_df["method"] == m].sort_values("kappa")
            ax.plot(sub["tps"], sub["em"], "-o", color=color, linewidth=2, markersize=8, label=label)
            for _, row in sub.iterrows():
                ax.annotate(f"κ={row['kappa']:g}", (row["tps"], row["em"]), textcoords="offset points", xytext=(6, 6), fontsize=9, color=color)
        ax.set_xlabel("Throughput (tokens/sec)")
        ax.set_ylabel("GSM8K exact match")
        ax.set_title("κ-sweep on 7B/1.5B — joint vs cascade traces")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = FIGS_DIR / "fig_bonus_kappa_sweep.pdf"
        fig.savefig(out_path)
        print("Saved:", out_path)
        plt.show()
'''
    )
)


CELLS.append(md_cell("## Theorem E — joint adaptivity beats fixed fuzzy threshold (14B/0.5B)\n\nAblation against `fuzzy_sd` (fixed γ=8, fixed T ∈ {1.0, 1.25, 1.5, 2.0}) on n=300. Joint dominates EVERY fixed-T baseline on BOTH EM and tok/s — adaptive control is empirically non-trivial vs any non-adaptive choice."))


CELLS.append(
    code_cell(
        '''fuzzy_csv = REPO_ROOT / "outputs" / "jointadaspec_qwen14b_0p5b_fuzzy_ablation_2026-05-25" / "03_bench_gsm8k" / "benchmark.csv"
if not fuzzy_csv.exists():
    print(f"[skip] {fuzzy_csv} not present")
else:
    df = load_benchmark(fuzzy_csv)
    summary = df.groupby("method").agg(em=("gsm8k_exact_match", "mean"),
                                       tps=("tokens_per_sec", "mean"),
                                       acc=("acceptance_rate", "mean")).reset_index()
    order = ["fuzzy_sd_T1", "fuzzy_sd_T1.25", "fuzzy_sd_T1.5", "fuzzy_sd_T2", "jointadaspec"]
    summary = summary.set_index("method").reindex(order).reset_index()
    print(summary.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#9aa0a6"] * 4 + ["#55A868"]
    axes[0].bar(range(len(summary)), summary["em"] * 100, color=colors)
    axes[0].set_xticks(range(len(summary)))
    axes[0].set_xticklabels(summary["method"], rotation=30, ha="right")
    axes[0].set_ylabel("GSM8K exact match (%)")
    axes[0].set_title("Quality")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(range(len(summary)), summary["tps"], color=colors)
    axes[1].set_xticks(range(len(summary)))
    axes[1].set_xticklabels(summary["method"], rotation=30, ha="right")
    axes[1].set_ylabel("Throughput (tokens/sec)")
    axes[1].set_title("Speed")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.suptitle("Theorem E — joint dominates fixed fuzzy_sd on quality AND speed (14B/0.5B, n=300)")
    fig.tight_layout()
    out_path = FIGS_DIR / "fig_E_adaptivity_ablation.pdf"
    fig.savefig(out_path)
    print("Saved:", out_path)
    plt.show()
'''
    )
)


CELLS.append(md_cell("## Theorem D — C4 violation is benign (replaces Theorem C's loose bound)\n\nC4 is violated on ~89% of states, but the cascade-policy *advantage* on those states is ≈0 → cascade is near-optimal in the visited regime. The exact stationary-weighted value gap V_joint − V_cascade is tiny on both pairs."))


CELLS.append(
    code_cell(
        '''import json
path = REPO_ROOT / "reports" / "theorem_c_gap_analysis.json"
if not path.exists():
    print(f"[skip] {path} not present")
else:
    data = json.loads(path.read_text())
    rows = [{"pair": k, **v} for k, v in data.items()]
    df = pd.DataFrame(rows)
    print(df[["pair", "c4_violating_state_frac", "mu_star_J_B",
              "theorem_c_bound", "exact_value_gap_stationary",
              "weak_dominance_frac_states"]].to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["mu_star_J_B"], width, label="μ*_J(B)  (C4-violation mass)", color="#C44E52")
    ax.bar(x + width / 2, df["exact_value_gap_stationary"] * 10, width, label="exact V-gap ×10", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["pair"])
    ax.set_title("Theorem D — pervasive C4 violation, near-zero realized gap")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = FIGS_DIR / "fig_D_advantage_on_B.pdf"
    fig.savefig(out_path)
    print("Saved:", out_path)
    plt.show()
'''
    )
)


CELLS.append(md_cell("## Theorem G — non-monotonic EM gain in joint acceptance (14B/0.5B)\n\nChurn rate ≈45% across all acceptance levels, but the NET EM change flips sign with acceptance: moderate acceptance helps, high acceptance hurts. Over-trusting the 0.5B draft degrades quality."))


CELLS.append(
    code_cell(
        '''lock = REPO_ROOT / "outputs" / "jointadaspec_qwen14b_0p5b_lock_2026-05-14" / "03_bench_gsm8k" / "benchmark.csv"
df = load_benchmark(lock)
w = df.pivot_table(index=["seed", "prompt_idx"], columns="method", values="gsm8k_exact_match")
acc = df[df["method"] == "jointadaspec"].set_index(["seed", "prompt_idx"])["acceptance_rate"]
d = pd.DataFrame({"t": w["target_only"], "j": w["jointadaspec"]}).join(acc.rename("acc")).dropna()
d["WR"] = ((d.t == 0) & (d.j == 1)).astype(int)
d["RW"] = ((d.t == 1) & (d.j == 0)).astype(int)
d["accbin"] = pd.qcut(d.acc, 3, labels=["low", "mid", "high"])
agg = d.groupby("accbin", observed=True).agg(
    WR=("WR", "mean"), RW=("RW", "mean"),
    n=("WR", "size"), accm=("acc", "mean"),
).reset_index()
agg["net"] = (agg.WR - agg.RW) * 100
print(agg.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#55A868" if v >= 0 else "#C44E52" for v in agg["net"]]
ax.bar(agg["accbin"].astype(str), agg["net"], color=colors)
for i, v in enumerate(agg["net"]):
    ax.text(i, v + (0.3 if v >= 0 else -0.5), f"{v:+.1f}%", ha="center", fontsize=11)
ax.set_xlabel("joint acceptance tercile")
ax.set_ylabel("net Δ EM (W→R minus R→W), %")
ax.set_title("Theorem G — quality gain non-monotonic in fuzzy acceptance")
ax.axhline(0, color="black", lw=0.5)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
out_path = FIGS_DIR / "fig_G_acceptance_em.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Policy interpretability — what did the joint controller learn?\n\nThree slices of the optimal 14B/0.5B policy: (1) P(continue drafting) vs entropy bin; (2) mean accept-threshold T* vs divergence bin; (3) P(continue) vs accepted-streak length k. Qualitatively the controller verifies stricter at high divergence and drafts maximally until the window cap."))


CELLS.append(
    code_cell(
        '''import sys
sys.path.insert(0, str(REPO_ROOT))
from jointadaspec.inference import JointAdaSpecPolicy
from jointadaspec.mdp.spaces import ActionSpace, StateSpace

pol = JointAdaSpecPolicy.load(str(REPO_ROOT / "outputs/jointadaspec_qwen14b_0p5b_2026-04-28/02_solve/policy.npz"))
cfg = pol.config
A = ActionSpace(cfg)
S = StateSpace(cfg)
contH = np.zeros(cfg.N_H); nH = np.zeros(cfg.N_H)
thrK = [[] for _ in range(cfg.N_K)]
cont_k = np.zeros(cfg.gamma_max + 1); n_k = np.zeros(cfg.gamma_max + 1)
for s in range(cfg.num_states):
    iH, iK, k = S.decode(s)
    a = A.decode(int(pol.pi_star[s]))
    c = 1.0 if a.length_action == "continue" else 0.0
    contH[iH] += c; nH[iH] += 1
    thrK[iK].append(a.threshold)
    cont_k[k] += c; n_k[k] += 1
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(range(cfg.N_H), contH / np.maximum(nH, 1) * 100, "-o", color="#4C72B0")
axes[0].set_xlabel("entropy bin i_H (low → high)")
axes[0].set_ylabel("P(continue), %")
axes[0].set_ylim(0, 105)
axes[0].set_title("Length action vs entropy")
axes[0].grid(True, alpha=0.3)
axes[1].plot(range(cfg.N_K), [np.mean(t) if t else np.nan for t in thrK], "-o", color="#C44E52")
axes[1].set_xlabel("divergence bin i_K (low → high)")
axes[1].set_ylabel("mean accept-threshold T*")
axes[1].set_title("Threshold vs K-divergence")
axes[1].grid(True, alpha=0.3)
axes[2].plot(range(cfg.gamma_max + 1), cont_k / np.maximum(n_k, 1) * 100, "-o", color="#55A868")
axes[2].set_xlabel("accepted-streak length k")
axes[2].set_ylabel("P(continue), %")
axes[2].set_ylim(0, 105)
axes[2].set_title("Length action vs streak")
axes[2].grid(True, alpha=0.3)
fig.suptitle("Learned joint policy on 14B/0.5B — mild adaptivity in qualitatively sensible directions")
fig.tight_layout()
out_path = FIGS_DIR / "fig_policy_interpretability.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## 7B/1.5B robustness — paired ΔEM across three held-out windows\n\nFirst window (n=200) showed +3.5% (noise at small n). At n=1500 on two fresh non-overlapping windows the effect is null/negative — joint=cascade hold, the method gives no quality benefit on this low-ratio (4.7×) pair."))


CELLS.append(
    code_cell(
        '''from scipy.stats import binomtest
windows = [
    ("start=1100\\nn=200", REPO_ROOT / "outputs/jointadaspec_qwen7b_1p5b_quality_k8_2026-05-12/03_bench_gsm8k/benchmark.csv"),
    ("start=100\\nn=1500 (lock)", REPO_ROOT / "outputs/jointadaspec_qwen7b_1p5b_quality_lock_2026-05-14/03_bench_gsm8k/benchmark.csv"),
    ("start=600\\nn=1500 (triang.)", REPO_ROOT / "outputs/jointadaspec_qwen7b_1p5b_quality_tri_2026-05-20/03_bench_gsm8k/benchmark.csv"),
]
rows = []
for label, p in windows:
    if not p.exists():
        print("[skip]", p); continue
    df = load_benchmark(p)
    w = df.pivot_table(index=["seed", "prompt_idx"], columns="method", values="gsm8k_exact_match")
    paired = w[["target_only", "jointadaspec"]].dropna()
    diff = (paired["jointadaspec"] - paired["target_only"]).mean() * 100
    wins = int(((paired["jointadaspec"] == 1) & (paired["target_only"] == 0)).sum())
    loss = int(((paired["jointadaspec"] == 0) & (paired["target_only"] == 1)).sum())
    pv = binomtest(min(wins, loss), wins + loss, 0.5).pvalue if (wins + loss) > 0 else 1.0
    rows.append({"window": label, "n": len(paired), "diff_pct": diff, "p": pv})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#55A868" if d >= 0 else "#C44E52" for d in out["diff_pct"]]
bars = ax.bar(out["window"], out["diff_pct"], color=colors)
for b, d, p in zip(bars, out["diff_pct"], out["p"]):
    ax.text(b.get_x() + b.get_width() / 2, d + (0.2 if d >= 0 else -0.5),
            f"{d:+.2f}%\\np={p:.3f}", ha="center", fontsize=10)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("paired ΔEM jointadaspec − target_only (%)")
ax.set_title("7B/1.5B — three independent held-out windows")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
out_path = FIGS_DIR / "fig_3win_robustness.pdf"
fig.savefig(out_path)
print("Saved:", out_path)
plt.show()
'''
    )
)


CELLS.append(md_cell("## Significance table for slides\n\nClean table dump for the slide deck — paste directly into the headline slide. Numbers are paired EM diff vs target_only, 95% bootstrap CI, and permutation p-value."))


CELLS.append(
    code_cell(
        '''def fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%"

table = diff_df.copy()
table["Δ EM"] = table["diff"].apply(fmt_pct)
table["95% CI"] = table.apply(lambda r: f"[{fmt_pct(r['ci_low'])}, {fmt_pct(r['ci_high'])}]", axis=1)
table["p"] = table["p_value"].apply(lambda p: f"{p:.4f}")
table["wins/losses/ties"] = table.apply(lambda r: f"{r['wins']}/{r['losses']}/{r['ties']}", axis=1)
display_cols = ["pair", "method", "n", "Δ EM", "95% CI", "p", "wins/losses/ties"]
print(table[display_cols].to_string(index=False))
'''
    )
)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    out_path = Path(__file__).resolve().parent / "thesis_plots.ipynb"
    with out_path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
