import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv, global_mean_pool

LOGGER = logging.getLogger("amc_ef")


def resolve_device(device=None, strict_cuda=False):
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved

    if not torch.cuda.is_available():
        return torch.device("cpu")

    index = 0 if resolved.index is None else resolved.index
    if index >= torch.cuda.device_count():
        if strict_cuda:
            raise RuntimeError(
                f"Requested CUDA device {index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        return torch.device("cuda:0")
    return torch.device(f"cuda:{index}")


def trees_to_graph(trees, num_node_types=13):
    all_node_features = []
    all_edge_index = []
    batch_indices = []
    node_id = 0

    def map_node_value_to_index(node_value):
        if isinstance(node_value, str):
            if node_value.startswith("-"):
                operator_idx = int(node_value[1:])
                if operator_idx < 0 or operator_idx > 4:
                    raise ValueError(f"Unsupported operator index: {node_value}")
                return 8 + operator_idx
            view_idx = int(node_value)
            if view_idx < 0 or view_idx > 7:
                raise ValueError(f"Unsupported view index: {node_value}")
            return view_idx

        node_value = int(node_value)
        if node_value >= 0:
            if node_value > 7:
                raise ValueError(f"Unsupported view index: {node_value}")
            return node_value
        if abs(node_value) > 4:
            raise ValueError(f"Unsupported operator index: {node_value}")
        return 8 + abs(node_value)

    def add_node(tree, node_tag, parent_id=None, batch_id=0):
        nonlocal node_id

        node = tree.nodes[node_tag]
        node_value = node.data if node.data is not None else node.tag

        try:
            index = map_node_value_to_index(node_value)
        except Exception as e:
            raise ValueError(f"Invalid node value for {node_tag}: {node_value}") from e

        node_feature = F.one_hot(torch.tensor(index), num_classes=num_node_types).float()

        current_id = node_id
        node_id += 1

        all_node_features.append(node_feature)
        batch_indices.append(batch_id)

        if parent_id is not None:
            all_edge_index.append([parent_id, current_id])
            all_edge_index.append([current_id, parent_id])

        for child in tree.children(node_tag):
            add_node(tree, child.identifier, current_id, batch_id)

    for batch_id, tree in enumerate(trees):
        add_node(tree, tree.root, batch_id=batch_id)

    x = torch.stack(all_node_features)
    if all_edge_index:
        edge_index = torch.tensor(all_edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.tensor(batch_indices, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, batch=batch)


class GCNPerformancePredictor(torch.nn.Module):
    def __init__(self, num_node_features=13, hidden_dim=64, num_heads=4, device=None):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim}, "
                f"num_heads={num_heads}"
            )

        self.device = resolve_device(device)
        out_channels = hidden_dim // num_heads
        self.conv1 = TransformerConv(
            in_channels=num_node_features,
            out_channels=out_channels,
            heads=num_heads,
            concat=True,
            dropout=0.1,
        )
        self.conv2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=out_channels,
            heads=num_heads,
            concat=True,
            dropout=0.1,
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.to(self.device)

    def forward(self, graph_data):
        if isinstance(graph_data, list):
            graph_data = Batch.from_data_list(graph_data)
        graph_data = graph_data.to(self.device)
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()

    def train_predictor(self, graph_data, true_performance, epochs=10):
        self.train()
        if isinstance(graph_data, list):
            graph_data = Batch.from_data_list(graph_data)
        if isinstance(true_performance, list):
            true_performance = torch.tensor(true_performance, dtype=torch.float)
        true_performance = true_performance.to(self.device)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred_performance = self.forward(graph_data)
            loss = self.criterion(pred_performance, true_performance)
            loss.backward()
            self.optimizer.step()
            LOGGER.debug(
                "predictor epoch %02d/%02d | loss=%.6f",
                epoch + 1,
                epochs,
                loss.item(),
            )

    def update_predictor(self, new_graph_data, new_true_performance, epochs=5):
        self.train_predictor(new_graph_data, new_true_performance, epochs)


def evaluate_predictor(predictor, graph_data, true_acc):
    predictor.eval()
    with torch.no_grad():
        pred_acc = predictor(graph_data)
        pred_acc = pred_acc.cpu().numpy()
        true_acc = np.array(true_acc)
        acc_mae = np.mean(np.abs(pred_acc - true_acc))
        acc_corr = np.corrcoef(pred_acc, true_acc)[0, 1] if len(pred_acc) > 1 else 0
        return {"acc_mae": acc_mae, "acc_corr": acc_corr}
