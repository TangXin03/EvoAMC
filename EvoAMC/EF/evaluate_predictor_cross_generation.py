import csv
import json
import os
import random
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import data_utils.config as config
from gcnpp import GCNPerformancePredictor, trees_to_graph
from utils_tree import list_to_tree


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


RESULT_CSV_PATH = os.environ.get("AMC_EF_RESULT_CSV_PATH") or None
HISTORY_CSV_PATH = os.environ.get("AMC_EF_HISTORY_CSV_PATH") or None
EPOCHS = 200
RANDOM_SEED = 42
DEVICE = None
EXCLUDE_SEEN_TEST_ARCHITECTURES = True


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


def resolve_path(explicit_path, filename):
    if explicit_path:
        if os.path.isabs(explicit_path):
            return explicit_path
        return os.path.join(SCRIPT_DIR, explicit_path)

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
                filename,
            )
        )

    result_save_dir = paras.get("result_save_dir")
    if result_save_dir:
        candidates.append(
            os.path.join(SCRIPT_DIR, data_name + "_view_result", result_save_dir, filename)
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    data_root = os.path.join(SCRIPT_DIR, data_name + "_view_result")
    fallback_candidates = sorted(glob(os.path.join(data_root, "**", filename), recursive=True))
    if len(fallback_candidates) == 1:
        return fallback_candidates[0]
    if len(fallback_candidates) > 1:
        candidate_list = "\n".join(fallback_candidates)
        raise FileNotFoundError(
            f"Resolved multiple {filename} files for the current data_name. "
            f"Please set the explicit path in evaluate_predictor_cross_generation.py.\n"
            f"{candidate_list}"
        )

    raise FileNotFoundError(
        f"Could not resolve {filename} from the current config. "
        f"Please set the explicit path in evaluate_predictor_cross_generation.py."
    )


def load_accuracy_map(result_csv_path):
    accuracy_map = {}
    with open(result_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            arch = row[0].strip()
            if not arch or arch in accuracy_map:
                continue
            accuracy_map[arch] = float(row[1])
    return accuracy_map


def load_generation_architectures(history_csv_path):
    generations = defaultdict(list)
    with open(history_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            generation = int(row[0])
            arch = row[1].strip()
            if arch:
                generations[generation].append(arch)

    ordered_generations = {}
    for generation in sorted(generations):
        seen = set()
        unique_archs = []
        for arch in generations[generation]:
            if arch in seen:
                continue
            seen.add(arch)
            unique_archs.append(arch)
        ordered_generations[generation] = unique_archs
    return ordered_generations


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
        return None

    rank_true = average_ranks(y_true)
    rank_pred = average_ranks(y_pred)
    if np.std(rank_true) == 0 or np.std(rank_pred) == 0:
        return None
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


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


def evaluate_gcn(train_pairs, test_pairs):
    train_graph, train_targets = build_graph_and_targets(train_pairs)
    test_graph, test_targets = build_graph_and_targets(test_pairs)

    predictor = GCNPerformancePredictor()
    train_targets = train_targets.to(predictor.device)

    for _ in range(EPOCHS):
        predictor.train()
        predictor.optimizer.zero_grad()
        train_pred = ensure_1d_tensor(predictor(train_graph))
        train_loss = predictor.criterion(train_pred, train_targets)
        train_loss.backward()
        predictor.optimizer.step()

    predictor.eval()
    with torch.no_grad():
        train_pred = ensure_1d_tensor(predictor(train_graph)).detach().cpu().numpy()
        test_pred = ensure_1d_tensor(predictor(test_graph)).detach().cpu().numpy()

    return {
        "train_spearman": spearman_corr(train_targets.detach().cpu().numpy(), train_pred),
        "test_spearman": spearman_corr(test_targets.detach().cpu().numpy(), test_pred),
    }


def evaluate_transformer(train_pairs, test_pairs):
    max_seq_len = get_max_sequence_len(train_pairs + test_pairs)
    train_tokens, train_mask, train_targets = build_sequence_batch(train_pairs, max_seq_len)
    test_tokens, test_mask, test_targets = build_sequence_batch(test_pairs, max_seq_len)

    predictor = TransformerPerformancePredictor(max_seq_len=max_seq_len, device=DEVICE)
    train_targets = train_targets.to(predictor.device)

    for _ in range(EPOCHS):
        predictor.train()
        predictor.optimizer.zero_grad()
        train_pred = predictor(train_tokens, train_mask)
        train_loss = predictor.criterion(train_pred, train_targets)
        train_loss.backward()
        predictor.optimizer.step()

    predictor.eval()
    with torch.no_grad():
        train_pred = predictor(train_tokens, train_mask).detach().cpu().numpy()
        test_pred = predictor(test_tokens, test_mask).detach().cpu().numpy()

    return {
        "train_spearman": spearman_corr(train_targets.detach().cpu().numpy(), train_pred),
        "test_spearman": spearman_corr(test_targets.detach().cpu().numpy(), test_pred),
    }


def save_results(output_dir, rows, metadata):
    metrics_csv_path = os.path.join(output_dir, "cross_generation_metrics.csv")
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "train_generations",
            "test_generation",
            "train_unique_count",
            "test_unique_count",
            "gcn_train_spearman",
            "gcn_test_spearman",
            "transformer_train_spearman",
            "transformer_test_spearman",
            "notes",
        ])
        for row in rows:
            writer.writerow([
                row["train_generations"],
                row["test_generation"],
                row["train_unique_count"],
                row["test_unique_count"],
                row["gcn_train_spearman"],
                row["gcn_test_spearman"],
                row["transformer_train_spearman"],
                row["transformer_test_spearman"],
                row["notes"],
            ])

    metrics_json_path = os.path.join(output_dir, "cross_generation_metrics.json")
    payload = {
        "metadata": metadata,
        "rows": rows,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return metrics_csv_path, metrics_json_path


def compute_summary(rows):
    def valid_values(key):
        return [row[key] for row in rows if row[key] is not None]

    summary = {
        "num_total_generations": len(rows),
        "num_valid_test_generations": len(valid_values("gcn_test_spearman")),
        "gcn_train_spearman_mean": None,
        "gcn_test_spearman_mean": None,
        "transformer_train_spearman_mean": None,
        "transformer_test_spearman_mean": None,
    }

    for key in [
        "gcn_train_spearman",
        "gcn_test_spearman",
        "transformer_train_spearman",
        "transformer_test_spearman",
    ]:
        values = valid_values(key)
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))

    return summary


def save_plot(output_dir, rows):
    valid_rows = [row for row in rows if row["gcn_test_spearman"] is not None and row["transformer_test_spearman"] is not None]
    if not valid_rows:
        return None

    generations = [row["test_generation"] for row in valid_rows]
    gcn_values = [row["gcn_test_spearman"] for row in valid_rows]
    transformer_values = [row["transformer_test_spearman"] for row in valid_rows]

    plot_path = os.path.join(output_dir, "cross_generation_spearman.png")
    plt.figure(figsize=(9, 5))
    plt.plot(generations, gcn_values, marker="o", label="GCN test Spearman")
    plt.plot(generations, transformer_values, marker="o", label="Transformer test Spearman")
    plt.xlabel("Test Generation")
    plt.ylabel("Spearman Correlation")
    plt.title("Cross-Generation Predictor Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def build_pairs_from_architectures(architectures, accuracy_map):
    pairs = []
    missing = []
    for arch in architectures:
        if arch not in accuracy_map:
            missing.append(arch)
            continue
        pairs.append((arch, accuracy_map[arch]))
    return pairs, missing


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    result_csv_path = resolve_path(RESULT_CSV_PATH, "result_KL.csv")
    history_csv_path = resolve_path(HISTORY_CSV_PATH, "history_n.csv")
    accuracy_map = load_accuracy_map(result_csv_path)
    generations = load_generation_architectures(history_csv_path)

    output_dir = os.path.join(
        os.path.dirname(result_csv_path),
        "predictor_cross_generation_compare",
    )
    os.makedirs(output_dir, exist_ok=True)

    cumulative_architectures = []
    cumulative_seen = set()
    rows = []
    generation_ids = sorted(generations)

    for generation_index in range(len(generation_ids) - 1):
        train_generation = generation_ids[generation_index]
        test_generation = generation_ids[generation_index + 1]

        for arch in generations[train_generation]:
            if arch in cumulative_seen:
                continue
            cumulative_seen.add(arch)
            cumulative_architectures.append(arch)

        test_architectures = list(generations[test_generation])
        if EXCLUDE_SEEN_TEST_ARCHITECTURES:
            test_architectures = [arch for arch in test_architectures if arch not in cumulative_seen]

        train_pairs, missing_train = build_pairs_from_architectures(cumulative_architectures, accuracy_map)
        test_pairs, missing_test = build_pairs_from_architectures(test_architectures, accuracy_map)

        notes = []
        if missing_train:
            notes.append(f"missing_train={len(missing_train)}")
        if missing_test:
            notes.append(f"missing_test={len(missing_test)}")

        row = {
            "train_generations": f"0-{train_generation}",
            "test_generation": test_generation,
            "train_unique_count": len(train_pairs),
            "test_unique_count": len(test_pairs),
            "gcn_train_spearman": None,
            "gcn_test_spearman": None,
            "transformer_train_spearman": None,
            "transformer_test_spearman": None,
            "notes": "",
        }

        if len(train_pairs) < 2:
            notes.append("skip_train_lt_2")
        if len(test_pairs) < 2:
            notes.append("skip_test_lt_2")

        if len(train_pairs) >= 2 and len(test_pairs) >= 2:
            gcn_metrics = evaluate_gcn(train_pairs, test_pairs)
            transformer_metrics = evaluate_transformer(train_pairs, test_pairs)
            row["gcn_train_spearman"] = gcn_metrics["train_spearman"]
            row["gcn_test_spearman"] = gcn_metrics["test_spearman"]
            row["transformer_train_spearman"] = transformer_metrics["train_spearman"]
            row["transformer_test_spearman"] = transformer_metrics["test_spearman"]

        row["notes"] = ";".join(notes)
        rows.append(row)
        print(
            f"train 0-{train_generation} -> test {test_generation} | "
            f"train={len(train_pairs)} | test={len(test_pairs)} | "
            f"gcn={row['gcn_test_spearman']} | transformer={row['transformer_test_spearman']} | "
            f"notes={row['notes']}"
        )

    metadata = {
        "result_csv_path": result_csv_path,
        "history_csv_path": history_csv_path,
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
        "exclude_seen_test_architectures": EXCLUDE_SEEN_TEST_ARCHITECTURES,
    }
    metrics_csv_path, metrics_json_path = save_results(output_dir, rows, metadata)
    plot_path = save_plot(output_dir, rows)
    summary = compute_summary(rows)

    summary_path = os.path.join(output_dir, "cross_generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics csv to: {metrics_csv_path}")
    print(f"Saved metrics json to: {metrics_json_path}")
    print(f"Saved summary to: {summary_path}")
    print(
        f"Average test Spearman | GCN={summary['gcn_test_spearman_mean']} | "
        f"Transformer={summary['transformer_test_spearman_mean']}"
    )
    if plot_path:
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
