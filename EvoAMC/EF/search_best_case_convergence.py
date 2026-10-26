import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_NAME = os.environ.get("AMC_EF_SEARCH_DATA_NAME", "rml2016_10a")
NB_ITERS = int(os.environ.get("AMC_EF_SEARCH_NB_ITERS", "30"))
RESULT_ROOT = os.environ.get(
    "AMC_EF_SEARCH_RESULT_ROOT",
    str(SCRIPT_DIR.parent / "results" / "convergence_search"),
)
USE_PARALLEL = os.environ.get("AMC_EF_SEARCH_USE_PARALLEL", "1") == "1"
TARGET_FINAL_GAP = float(os.environ.get("AMC_EF_SEARCH_TARGET_FINAL_GAP", "0.01"))
TARGET_CONSECUTIVE_GENS = int(os.environ.get("AMC_EF_SEARCH_TARGET_CONSECUTIVE_GENS", "5"))
DEFAULT_SEARCH_SNRS = [
    "view_snr-4",
    "view_snr-8",
    "view_snr-12",
    "view_snr-16",
    "view_snr-20",
]


def parse_int_list(value, default):
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def data_root():
    return Path(os.environ.get("AMC_EF_DATA_ROOT", SCRIPT_DIR.parent / "data"))


def search_seeds():
    return parse_int_list(os.environ.get("AMC_EF_SEARCH_SEEDS"), [0, 1, 2, 3, 4])


def discover_snrs():
    configured = parse_str_list(os.environ.get("AMC_EF_SEARCH_SNRS"))
    snr_names = configured if configured else list(DEFAULT_SEARCH_SNRS)

    dataset_dir = data_root() / DATA_NAME
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    available = {path.name for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith("view_snr")}
    if not available:
        raise FileNotFoundError(f"No view_snr* directories found under: {dataset_dir}")

    missing = [snr_name for snr_name in snr_names if snr_name not in available]
    if missing:
        raise FileNotFoundError(f"Requested SNR directories not found under {dataset_dir}: {missing}")
    return snr_names


def entry_script():
    return SCRIPT_DIR / ("train_convergence_compare_parallel.py" if USE_PARALLEL else "train_convergence_compare.py")


def effective_result_root():
    if USE_PARALLEL:
        return os.environ.get(
            "AMC_EF_CONVERGENCE_PARALLEL_RESULT_ROOT",
            RESULT_ROOT,
        )
    return RESULT_ROOT


def run_case(snr_name, seed):
    env = os.environ.copy()
    env["AMC_EF_CC_DATA_NAME"] = DATA_NAME
    env["AMC_EF_CC_SNR_NAME"] = snr_name
    env["AMC_EF_CC_NB_ITERS"] = str(NB_ITERS)
    env["AMC_EF_CC_SEEDS"] = str(seed)
    env["AMC_EF_CC_RESULT_ROOT"] = RESULT_ROOT
    if USE_PARALLEL:
        env["AMC_EF_CONVERGENCE_PARALLEL_RESULT_ROOT"] = RESULT_ROOT

    command = [sys.executable, str(entry_script())]
    print(f"Running case | data={DATA_NAME} | snr={snr_name} | seed={seed} | parallel={USE_PARALLEL}")
    subprocess.run(command, cwd=str(SCRIPT_DIR), env=env, check=True)


def summary_csv_path():
    return SCRIPT_DIR / effective_result_root() / DATA_NAME / "all_best_case_dummy"


def load_case_curve(snr_name, seed):
    csv_path = SCRIPT_DIR / effective_result_root() / DATA_NAME / snr_name / "all_convergence_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary csv not found after run: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["seed"] == seed].copy()
    if df.empty:
        raise ValueError(f"No rows found for seed={seed} in {csv_path}")

    ours = df[df["method"] == "ours_multi_offspring_knowledge"][["generation", "population_best_fitness"]].copy()
    baseline = df[df["method"] == "baseline_single_offspring"][["generation", "population_best_fitness"]].copy()
    merged = ours.merge(
        baseline,
        on="generation",
        suffixes=("_ours", "_baseline"),
        how="inner",
    ).sort_values("generation")
    if merged.empty:
        raise ValueError(f"No overlapping generations found for seed={seed} in {csv_path}")
    merged["gap"] = merged["population_best_fitness_ours"] - merged["population_best_fitness_baseline"]
    return csv_path, merged


def score_case(curve_df):
    gap = curve_df["gap"]
    mean_gap = float(gap.mean())
    final_gap = float(gap.iloc[-1])
    min_gap = float(gap.min())
    max_consecutive_gap = longest_consecutive_run(gap >= TARGET_FINAL_GAP)
    score = mean_gap + 0.5 * final_gap + 0.25 * min_gap
    return {
        "score": score,
        "mean_gap": mean_gap,
        "final_gap": final_gap,
        "min_gap": min_gap,
        "max_consecutive_gap": max_consecutive_gap,
    }


def longest_consecutive_run(mask):
    best = 0
    current = 0
    for flag in mask.tolist():
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return int(best)


def reached_target(case):
    return case["max_consecutive_gap"] >= TARGET_CONSECUTIVE_GENS


def plot_best_case(best_case):
    if plt is None:
        print("matplotlib is not installed; skip plotting")
        return None

    curve_df = best_case["curve_df"]
    snr_name = best_case["snr_name"]
    seed = best_case["seed"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.3,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "lines.markersize": 4.8,
        }
    )

    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    styles = {
        "ours": {"label": "AMC-EvoMVF", "color": "#D55E00", "marker": "o", "linestyle": "-"},
        "baseline": {"label": "Baseline", "color": "#0072B2", "marker": "s", "linestyle": "--"},
    }

    for key, column in [("ours", "population_best_fitness_ours"), ("baseline", "population_best_fitness_baseline")]:
        style = styles[key]
        y_values = curve_df[column] * 100.0
        markevery = max(1, len(curve_df) // 10)
        ax.plot(
            curve_df["generation"],
            y_values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=markevery,
            markerfacecolor="white",
            markeredgewidth=1.3,
        )
        final_x = int(curve_df["generation"].iloc[-1])
        final_y = float(y_values.iloc[-1])
        ax.scatter([final_x], [final_y], color=style["color"], s=36, zorder=5)
        ax.annotate(
            f"{final_y:.2f}",
            xy=(final_x, final_y),
            xytext=(-4, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.7,
            color=style["color"],
        )

    from matplotlib.ticker import MaxNLocator

    ax.set_title("Exploratory Best-Case Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Individual Accuracy (%)")
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_dir = SCRIPT_DIR / effective_result_root() / DATA_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "exploratory_best_case_curve.pdf"
    png_path = output_dir / "exploratory_best_case_curve.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    return pdf_path, png_path


def write_summary(best_case, all_cases):
    output_dir = SCRIPT_DIR / effective_result_root() / DATA_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "exploratory_best_case_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Exploratory best-case search summary\n")
        f.write("This result is selected from an SNR/seed sweep and should not be presented as an aggregated result.\n\n")
        f.write(f"data_name: {DATA_NAME}\n")
        f.write(f"nb_iters: {NB_ITERS}\n")
        f.write(f"use_parallel: {USE_PARALLEL}\n")
        f.write(f"target_gap: {TARGET_FINAL_GAP:.6f}\n")
        f.write(f"target_consecutive_generations: {TARGET_CONSECUTIVE_GENS}\n")
        f.write(f"best_snr: {best_case['snr_name']}\n")
        f.write(f"best_seed: {best_case['seed']}\n")
        f.write(f"score: {best_case['score']:.6f}\n")
        f.write(f"mean_gap: {best_case['mean_gap']:.6f}\n")
        f.write(f"final_gap: {best_case['final_gap']:.6f}\n")
        f.write(f"min_gap: {best_case['min_gap']:.6f}\n")
        f.write(f"max_consecutive_gap: {best_case['max_consecutive_gap']}\n")
        f.write(f"summary_csv: {best_case['csv_path']}\n\n")
        f.write("All cases:\n")
        for case in all_cases:
            f.write(
                f"snr={case['snr_name']}, seed={case['seed']}, score={case['score']:.6f}, "
                f"mean_gap={case['mean_gap']:.6f}, final_gap={case['final_gap']:.6f}, "
                f"min_gap={case['min_gap']:.6f}, max_consecutive_gap={case['max_consecutive_gap']}\n"
            )
    print(f"Saved summary: {summary_path}")


def main():
    snr_names = discover_snrs()
    seeds = search_seeds()
    all_cases = []
    stop_case = None

    for snr_name in snr_names:
        for seed in seeds:
            run_case(snr_name, seed)
            csv_path, curve_df = load_case_curve(snr_name, seed)
            metrics = score_case(curve_df)
            case = {
                "snr_name": snr_name,
                "seed": seed,
                "csv_path": csv_path,
                "curve_df": curve_df,
                **metrics,
            }
            all_cases.append(case)
            print(
                f"Case done | snr={snr_name} | seed={seed} | "
                f"score={case['score']:.6f} | mean_gap={case['mean_gap']:.6f} | "
                f"final_gap={case['final_gap']:.6f} | min_gap={case['min_gap']:.6f} | "
                f"max_consecutive_gap={case['max_consecutive_gap']}"
            )
            if reached_target(case):
                stop_case = case
                print(
                    f"Early stop triggered | snr={snr_name} | seed={seed} | "
                    f"max_consecutive_gap={case['max_consecutive_gap']} >= "
                    f"target_consecutive_generations={TARGET_CONSECUTIVE_GENS} "
                    f"with gap threshold={TARGET_FINAL_GAP:.6f}"
                )
                break
        if stop_case is not None:
            break

    if not all_cases:
        raise RuntimeError("No cases were executed.")

    best_case = stop_case if stop_case is not None else max(all_cases, key=lambda case: case["score"])
    plot_best_case(best_case)
    write_summary(best_case, sorted(all_cases, key=lambda case: case["score"], reverse=True))


if __name__ == "__main__":
    main()
