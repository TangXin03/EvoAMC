import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import json
import os
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
import torch.nn as nn

import data_utils.config as config


# Set this to an absolute/relative result_KL.csv path if you do not want auto-resolution.
RESULT_CSV_PATH = os.environ.get("AMC_EF_RESULT_CSV_PATH") or None
# Number of architecture-accuracy pairs used for training.
TRAIN_SIZE = 32
# Predictor training epochs for the holdout experiment.
EPOCHS = 150
# Mark and report train/test Spearman on the loss curve every N epochs.
SPEARMAN_MARK_INTERVAL = 30
# Random seed used for shuffling the architecture-accuracy pairs.
RANDOM_SEED = 0
# Set to "cpu" or "cuda:0" to override auto device selection.
DEVICE = None


paras = config.get_configs()
SCRIPT_DIR = paras.get("result_root", os.path.dirname(os.path.abspath(__file__)))

PAD_ID = 13
TOKEN_TO_ID = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "-0": 8,
    "-1": 9,
    "-2": 10,
    "-3": 11,
    "-4": 12,
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device=None):
    if device is not None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return resolved
        if not torch.cuda.is_available():
            return torch.device("cpu")
        index = 0 if resolved.index is None else resolved.index
        if index >= torch.cuda.device_count():
            return torch.device("cuda:0")
        return torch.device(f"cuda:{index}")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda:0")
    return torch.device("cpu")


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
            "Please set RESULT_CSV_PATH explicitly in evaluate_predictor_holdout_transformer.py.\n"
            f"{candidate_list}"
        )

    raise FileNotFoundError(
        "Could not resolve result_KL.csv from the current config. "
        "Please set RESULT_CSV_PATH explicitly in evaluate_predictor_holdout_transformer.py."
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


def encode_architecture(arch):
    token_ids = []
    for token in arch.split("+"):
        if token not in TOKEN_TO_ID:
            raise ValueError(f"Unsupported architecture token: {token}")
        token_ids.append(TOKEN_TO_ID[token])
    return token_ids


def get_max_sequence_len(pairs):
    return max(len(encode_architecture(arch)) for arch, _ in pairs)


def build_sequence_batch(pairs, max_seq_len):
    token_tensor = torch.full((len(pairs), max_seq_len), PAD_ID, dtype=torch.long)
    padding_mask = torch.ones((len(pairs), max_seq_len), dtype=torch.bool)
    targets = []

    for row_idx, (arch, acc) in enumerate(pairs):
        token_ids = encode_architecture(arch)
        token_tensor[row_idx, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        padding_mask[row_idx, :len(token_ids)] = False
        targets.append(acc)

    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return token_tensor, padding_mask, target_tensor


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


class TransformerPerformancePredictor(nn.Module):
    def __init__(self, max_seq_len, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, device=None):
        super().__init__()
        self.device = resolve_device(device)
        self.token_embedding = nn.Embedding(PAD_ID + 1, model_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.to(self.device)

    def forward(self, token_ids, padding_mask):
        token_ids = token_ids.to(self.device)
        padding_mask = padding_mask.to(self.device)
        positions = torch.arange(token_ids.size(1), device=self.device).unsqueeze(0)
        positions = positions.expand(token_ids.size(0), token_ids.size(1))

        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        return self.regressor(pooled).squeeze(-1)


def train_predictor_with_history(train_tokens, train_mask, train_targets, test_tokens, test_mask, test_targets, max_seq_len):
    predictor = TransformerPerformancePredictor(max_seq_len=max_seq_len, device=DEVICE)
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
        train_pred = predictor(train_tokens, train_mask)
        train_loss = predictor.criterion(train_pred, train_targets)
        train_loss.backward()
        predictor.optimizer.step()

        predictor.eval()
        with torch.no_grad():
            test_pred = predictor(test_tokens, test_mask)
            test_loss = predictor.criterion(test_pred, test_targets)
            if should_mark_spearman(epoch):
                train_pred_eval = predictor(train_tokens, train_mask)
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


def evaluate_correlation(predictor, token_ids, padding_mask, targets):
    predictor.eval()
    with torch.no_grad():
        pred = predictor(token_ids, padding_mask).detach().cpu().numpy()
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
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    test_loss,
                    "" if train_corr is None else train_corr,
                    "" if test_corr is None else test_corr,
                ]
            )
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
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
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
            xytext = (-8, label_offset)
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
                fontsize=9,
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

    ax.set_xlabel("Training Epoch", fontsize=13)
    ax.set_ylabel("Mean Squared Error", fontsize=13)
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
        "model_type": "transformer",
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
    max_seq_len = get_max_sequence_len(train_pairs + test_pairs)

    train_tokens, train_mask, train_targets = build_sequence_batch(train_pairs, max_seq_len)
    test_tokens, test_mask, test_targets = build_sequence_batch(test_pairs, max_seq_len)

    predictor, history = train_predictor_with_history(
        train_tokens,
        train_mask,
        train_targets,
        test_tokens,
        test_mask,
        test_targets,
        max_seq_len,
    )

    train_corr, _ = evaluate_correlation(predictor, train_tokens, train_mask, train_targets)
    test_corr, _ = evaluate_correlation(predictor, test_tokens, test_mask, test_targets)

    output_dir = os.path.join(
        os.path.dirname(result_csv_path),
        f"predictor_holdout_transformer_train_{len(train_pairs)}",
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
