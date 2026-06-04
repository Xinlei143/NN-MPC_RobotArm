from __future__ import annotations

import torch
from torch import nn


class MLPDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256, output_dim: int | None = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim if output_dim is None else output_dim
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MLPDynamics expects input shape [batch, input_dim], got {tuple(x.shape)}")
        return self.net(x)


class GRUDynamics(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim if output_dim is None else output_dim
        input_dim = state_dim + action_dim
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"GRUDynamics expects input shape [batch, history_len, input_dim], got {tuple(x.shape)}")
        output, _ = self.gru(x)
        return self.head(output[:, -1])


class TransformerDynamics(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_history_len: int = 256,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim if output_dim is None else output_dim
        self.max_history_len = max_history_len
        input_dim = state_dim + action_dim
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_history_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"TransformerDynamics expects input shape [batch, history_len, input_dim], got {tuple(x.shape)}"
            )
        history_len = x.shape[1]
        if history_len > self.max_history_len:
            raise ValueError(f"history_len={history_len} exceeds max_history_len={self.max_history_len}")
        tokens = self.embedding(x) + self.pos_embedding[:, :history_len]
        encoded = self.encoder(tokens)
        return self.head(encoded[:, -1])
