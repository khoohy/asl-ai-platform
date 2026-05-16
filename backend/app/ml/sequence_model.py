"""Production BiLSTM classifier for the 30 x 180 ASL sequence model."""

import torch
import torch.nn as nn


class BiLSTMSignClassifier(nn.Module):
    """Production-aligned BiLSTM with attention pooling for WLASL300."""

    def __init__(
        self,
        input_dim: int = 180,
        hidden_dim: int = 512,
        num_classes: int = 300,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_heads: int = 1,
        use_motion_delta: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_motion_delta = use_motion_delta

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        frame_weights = torch.softmax(self.attention_pool(lstm_out), dim=1)
        pooled = torch.sum(lstm_out * frame_weights, dim=1)
        return self.classifier(pooled)
