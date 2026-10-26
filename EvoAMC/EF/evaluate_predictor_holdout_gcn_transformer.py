import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import json
import random
from glob import glob

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
import torch

import data_utils.config as config
from gcn_transformer import GCNTransformerPerformancePredictor, trees_to_graph
from utils_tree import list_to_tree


# Set this to an absolute/relative result_KL.csv path if you do not want auto-resolution.
RESULT_CSV_PATH = os.environ.get("AMC_EF_RESULT_CSV_PATH") or None
# Number of architecture-accuracy pairs used for training.
TRAIN_SIZE = 32
# Predictor training epochs for the holdout experiment.
EPOCHS = 150
# Mark and report train/test Spearman on the loss curve every N epochs.
SPEARMAN_MARK_INTERVAL = 30
# Random seed used for shuffling the architecture-accuracy pairs.
RANDOM_SEED = 6  #6
# Set to "cpu" or "cuda:0" to override auto device selection.
DEVICE = None


paras = config.get_configs()
SCRIPT_DIR = paras.get("result_root", os.path.dirname(os.path.abspath(__file__)))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_result_csv_path():
    if RESULT_CSV_PATH:
        if os.path.isabs(RESULT_CSV_PATH):
            return RESULT_CSV_PATH
        return os.path.join(SCRIPT_DIR, RESULT_CSV_PATH)

    data_name = paras["data_name"]
    candidates = []

    ablation_name = paras.get("ablation_name")
    snr_name = paras.get("snr_name")
    if ablation_name and snr_name:
        candidates.append(
            os.path.join(
                SCRIPT_DIR,
                data_name + "_view_result",
                "ablation",
                ablation_name,
                snr_name,
                "result_KL.csv",
            )
        )

    result_save_dir = paras.get("result_save_dir")
    if result_save_dir:
        if snr_name:
            candidates.append(
                os.path.join(SCRIPT_DIR, data_name + "_view_result", snr_name, result_save_dir, "result_KL.csv")
            )
        candidates.append(
            os.path.join(SCRIPT_DIR, data_name + "_view_result", result_save_dir, "result_KL.csv")
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    data_root = os.path.join(SCRIPT_DIR, data_name + "_view_result")
    fallback_candidates = sorted(glob(os.path.join(data_root, "**", "result_KL.csv"), recursive=True))
    if len(fallback_candidates) == 1:
        return fallback_candidates[0]
    if len(fallback_candidates) > 1:
        candidate_list = "\n".join(fallback_candidates)
        raise FileNotFoundError(
            "Resolved multiple result_KL.csv files for the current data_name. "
            "Please set RESULT_CSV_PATH explicitly in evaluate_predictor_holdout_gcn_transformer.py.\n"
            f"{candidate_list}"
        )

    raise FileNotFoundError(
        "Could not resolve result_KL.csv from the current config. "
        "Please set RESULT_CSV_PATH explicitly in evaluate_predictor_holdout_gcn_transformer.py."
    )


def load_architecture_accuracy_pairs(result_csv_path):
    pairs = []
    seen = set()
    with open(result_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            arch = row[0].strip()
            if not arch or arch in seen:
                continue
            acc = float(row[1])
            pairs.append((arch, acc))
            seen.add(arch)
    return pairs


def split_pairs(pairs, train_size, seed):
    if train_size <= 0:
        raise ValueError(f"TRAIN_SIZE must be positive, got {train_size}.")
    if train_size >= len(pairs):
        raise ValueError(
            f"TRAIN_SIZE must be smaller than the total number of samples. "
            f"train_size={train_size}, total={len(pairs)}"
        )

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_size], shuffled[train_size:]


def build_graph_and_targets(pairs):
    trees = []
    targets = []
    for arch, acc in pairs:
        trees.append(list_to_tree(arch.split("+")))
        targets.append(acc)

    graph = trees_to_graph(trees)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return graph, target_tensor


def ensure_1d_tensor(tensor):
    return tensor.unsqueeze(0) if tensor.dim() == 0 else tensor


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1

        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end

    return ranks


def spearman_corr(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0

    rank_true = average_ranks(y_true)
    rank_pred = average_ranks(y_pred)
    if np.std(rank_true) == 0 or np.std(rank_pred) == 0:
        return 0.0
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


def should_mark_spearman(epoch):
    return epoch % SPEARMAN_MARK_INTERVAL == 0 or epoch == EPOCHS


def train_predictor_with_history(train_graph, train_targets, test_graph, test_targets):
    predictor = GCNTransformerPerformancePredictor(device=DEVICE)
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_spearman": [],
        "test_spearman": [],
    }

    train_targets = train_targets.to(predictor.device)
    test_targets = test_targets.to(predictor.device)

    for epoch in range(1, EPOCHS + 1):
        predictor.train()
        predictor.optimizer.zero_grad()
        train_pred = ensure_1d_tensor(predictor(train_graph))
        train_loss = predictor.criterion(train_pred, train_targets)
        train_loss.backward()
        predictor.optimizer.step()

        predictor.eval()
        with torch.no_grad():
            test_pred = ensure_1d_tensor(predictor(test_graph))
            test_loss = predictor.criterion(test_pred, test_targets)
            if should_mark_spearman(epoch):
                train_pred_eval = ensure_1d_tensor(predictor(train_graph))
                train_corr = spearman_corr(
                    train_targets.detach().cpu().numpy(),
                    train_pred_eval.detach().cpu().numpy(),
                )
                test_corr = spearman_corr(
                    test_targets.detach().cpu().numpy(),
                    test_pred.detach().cpu().numpy(),
                )
            else:
                train_corr = None
                test_corr = None

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss.item()))
        history["test_loss"].append(float(test_loss.item()))
        history["train_spearman"].append(train_corr)
        history["test_spearman"].append(test_corr)

    return predictor, history


def evaluate_correlation(predictor, graph, targets):
    predictor.eval()
    with torch.no_grad():
        pred = ensure_1d_tensor(predictor(graph)).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    return spearman_corr(true, pred), pred


def save_loss_history(output_dir, history):
    history_csv = os.path.join(output_dir, "loss_history.csv")
    with open(history_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "train_spearman", "test_spearman"])
        for epoch, train_loss, test_loss, train_corr, test_corr in zip(
            history["epoch"],
            history["train_loss"],
            history["test_loss"],
            history["train_spearman"],
            history["test_spearman"],
        ):
            writer.writerow([
                epoch,
                train_loss,
                test_loss,
                "" if train_corr is None else train_corr,
                "" if test_corr is None else test_corr,
            ])
    return history_csv


def save_loss_curve(output_dir, history):
    if plt is None:
        return None

    curve_path = os.path.join(output_dir, "loss_curve.png")
    plt.rcParams.update({
        "font.family": "Calibri",
        "font.size": 12,
        "axes.linewidth": 1.1,
        "figure.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        history["epoch"],
        history["train_loss"],
        linewidth=1.5,
        color="#0072B2",
        label="Training loss",
    )
    ax.plot(
        history["epoch"],
        history["test_loss"],
        linewidth=1.5,
        color="#D55E00",
        label="Test loss",
    )

    marked = [
        idx
        for idx, (train_corr, test_corr) in enumerate(
            zip(history["train_spearman"], history["test_spearman"])
        )
        if train_corr is not None and test_corr is not None
    ]
    if marked:
        marked_epochs = [history["epoch"][idx] for idx in marked]
        marked_train_loss = [history["train_loss"][idx] for idx in marked]
        marked_test_loss = [history["test_loss"][idx] for idx in marked]

        ax.scatter(
            marked_epochs,
            marked_test_loss,
            s=48,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
            label="Spearman checkpoints",
        )

        label_offset = 10
        for mark_i, idx in enumerate(marked):
            xytext = (-12, label_offset)
            ax.annotate(
                (
                    f"Train: {history['train_spearman'][idx]:.3f}\n"
                    f"Test: {history['test_spearman'][idx]:.3f}"
                ),
                xy=(history["epoch"][idx], history["test_loss"][idx]),
                xytext=xytext,
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                color="#202020",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#808080",
                    "alpha": 0.96,
                },
                zorder=7,
                clip_on=False,
            )

    ax.set_xlabel("Training Epoch", fontsize=14)
    ax.set_ylabel("Mean Squared Error", fontsize=14)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=12, ncol=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.96)
    plt.savefig(curve_path, dpi=600, bbox_inches="tight")
    plt.close()
    return curve_path


def save_metrics(output_dir, result_csv_path, train_pairs, test_pairs, train_corr, test_corr, device):
    metrics = {
        "model_type": "gcn_transformer",
        "result_csv_path": result_csv_path,
        "train_size": len(train_pairs),
        "test_size": len(test_pairs),
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "train_spearman_corr": train_corr,
        "test_spearman_corr": test_corr,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path, metrics


def main():
    set_random_seed(RANDOM_SEED)
    result_csv_path = resolve_result_csv_path()
    pairs = load_architecture_accuracy_pairs(result_csv_path)
    train_pairs, test_pairs = split_pairs(pairs, TRAIN_SIZE, RANDOM_SEED)

    train_graph, train_targets = build_graph_and_targets(train_pairs)
    test_graph, test_targets = build_graph_and_targets(test_pairs)

    predictor, history = train_predictor_with_history(
        train_graph,
        train_targets,
        test_graph,
        test_targets,
    )

    train_corr, _ = evaluate_correlation(predictor, train_graph, train_targets)
    test_corr, _ = evaluate_correlation(predictor, test_graph, test_targets)

    output_dir = os.path.join(
        os.path.dirname(result_csv_path),
        f"predictor_holdout_gcn_transformer_train_{len(train_pairs)}",
    )
    os.makedirs(output_dir, exist_ok=True)

    history_csv = save_loss_history(output_dir, history)
    curve_path = save_loss_curve(output_dir, history)
    metrics_path, metrics = save_metrics(
        output_dir,
        result_csv_path,
        train_pairs,
        test_pairs,
        train_corr,
        test_corr,
        predictor.device,
    )

    print(f"Result CSV: {result_csv_path}")
    print(f"Training samples: {len(train_pairs)}")
    print(f"Test samples: {len(test_pairs)}")
    print(f"Device: {predictor.device}")
    print(f"Train Spearman corr: {metrics['train_spearman_corr']:.6f}")
    print(f"Test Spearman corr: {metrics['test_spearman_corr']:.6f}")
    print(f"Saved loss history to: {history_csv}")
    if curve_path is None:
        print("Skipped loss curve: matplotlib is not installed")
    else:
        print(f"Saved loss curve to: {curve_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
