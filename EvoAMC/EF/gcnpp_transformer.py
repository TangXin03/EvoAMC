import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, global_mean_pool

from gcnpp import resolve_device, trees_to_graph

LOGGER = logging.getLogger("amc_ef")


class TransformerPerformancePredictor(torch.nn.Module):
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
                "transformer predictor epoch %02d/%02d | loss=%.6f",
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


__all__ = [
    "TransformerPerformancePredictor",
    "evaluate_predictor",
    "resolve_device",
    "trees_to_graph",
]
