"""Model definitions: per-frame eye-state CNN and temporal drowsiness LSTM."""
import torch
import torch.nn as nn


class EyeStateCNN(nn.Module):
    """Classifies a single eye crop as open (0) or closed (1).
    Also exposes `embed()` to get a feature vector for the LSTM stage."""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed_fc = nn.Linear(64, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def embed(self, x):
        x = self.features(x).flatten(1)
        return torch.relu(self.embed_fc(x))

    def forward(self, x):
        emb = self.embed(x)
        return self.classifier(emb).squeeze(-1)


class DrowsinessLSTM(nn.Module):
    """Takes a sequence of per-frame feature vectors and predicts drowsy/alert
    for the whole window. feature_dim=3 for the lightweight [p_closed, EAR, MAR]
    input; use a larger feature_dim if concatenating CNN embeddings."""

    def __init__(self, feature_dim=3, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)
