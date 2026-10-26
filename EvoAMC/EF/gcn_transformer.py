import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

from gcnpp import resolve_device, trees_to_graph


class GCNTransformerPerformancePredictor(nn.Module):
    def __init__(
        self,
        num_node_features=13,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.15,
        device=None,
    ):
        super().__init__()
        self.device = resolve_device(device)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=8e-4, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.to(self.device)

    def forward(self, graph_data):
        if isinstance(graph_data, list):
            graph_data = Batch.from_data_list(graph_data)
        graph_data = graph_data.to(self.device)

        x = F.relu(self.conv1(graph_data.x, graph_data.edge_index))
        x = F.relu(self.conv2(x, graph_data.edge_index))
        x = self.input_norm(x)

        dense_x, mask = to_dense_batch(x, graph_data.batch)
        encoded = self.transformer(dense_x, src_key_padding_mask=~mask)

        mask_float = mask.unsqueeze(-1).float()
        graph_embedding = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        prediction = self.head(graph_embedding)
        return prediction.squeeze(-1)

    def train_predictor(self, graph_data, true_performance, epochs=10):
        self.train()
        if isinstance(graph_data, list):
            graph_data = Batch.from_data_list(graph_data)
        if isinstance(true_performance, list):
            true_performance = torch.tensor(true_performance, dtype=torch.float32)
        true_performance = true_performance.to(self.device)

        for _ in range(epochs):
            self.optimizer.zero_grad()
            prediction = self.forward(graph_data)
            loss = self.criterion(prediction, true_performance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def update_predictor(self, new_graph_data, new_true_performance, epochs=5):
        self.train_predictor(new_graph_data, new_true_performance, epochs)
