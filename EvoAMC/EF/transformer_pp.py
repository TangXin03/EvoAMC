import torch
import torch.nn as nn

from gcnpp import resolve_device


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


def encode_architecture(individual):
    token_ids = []
    for token in individual:
        token = str(token)
        if token not in TOKEN_TO_ID:
            raise ValueError(f"Unsupported architecture token: {token}")
        token_ids.append(TOKEN_TO_ID[token])
    return token_ids


def architectures_to_sequence(individuals, max_seq_len=None):
    encoded = [encode_architecture(individual) for individual in individuals]
    if not encoded:
        raise ValueError("individuals must not be empty.")

    if max_seq_len is None:
        max_seq_len = max(len(tokens) for tokens in encoded)
    if max(len(tokens) for tokens in encoded) > max_seq_len:
        raise ValueError(f"Architecture length exceeds max_seq_len={max_seq_len}.")

    token_tensor = torch.full((len(encoded), max_seq_len), PAD_ID, dtype=torch.long)
    padding_mask = torch.ones((len(encoded), max_seq_len), dtype=torch.bool)
    for row_idx, token_ids in enumerate(encoded):
        length = len(token_ids)
        token_tensor[row_idx, :length] = torch.tensor(token_ids, dtype=torch.long)
        padding_mask[row_idx, :length] = False
    return token_tensor, padding_mask


class TransformerPerformancePredictor(nn.Module):
    def __init__(
        self,
        max_seq_len=40,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        device=None,
    ):
        super().__init__()
        self.device = resolve_device(device)
        self.max_seq_len = max_seq_len
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

    def forward(self, sequence_data):
        token_ids, padding_mask = sequence_data
        token_ids = token_ids.to(self.device)
        padding_mask = padding_mask.to(self.device)

        positions = torch.arange(token_ids.size(1), device=self.device).unsqueeze(0)
        positions = positions.expand(token_ids.size(0), token_ids.size(1))

        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        return self.regressor(pooled).squeeze(-1)

    def train_predictor(self, sequence_data, true_performance, epochs=10):
        self.train()
        if isinstance(true_performance, list):
            true_performance = torch.tensor(true_performance, dtype=torch.float32)
        true_performance = true_performance.to(self.device)

        for _ in range(epochs):
            self.optimizer.zero_grad()
            prediction = self.forward(sequence_data)
            loss = self.criterion(prediction, true_performance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def update_predictor(self, new_sequence_data, new_true_performance, epochs=5):
        self.train_predictor(new_sequence_data, new_true_performance, epochs)
