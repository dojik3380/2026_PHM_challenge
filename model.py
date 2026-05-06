"""STFT + CNN + LSTM 멀티브랜치 RUL 모델."""

import math

import numpy as np
import torch
import torch.nn as nn

from config import DROPOUT


class STFTCNNLSTMRULModel(nn.Module):
    """
    진동 branch: STFT 입력 -> CNN -> LSTM
    운전 branch: 운전 시퀀스 -> LSTM
    Fusion head: 두 branch 출력을 결합해 RUL 예측
    """

    def __init__(
        self,
        vibration_channels: int = 4,
        operation_features: int = 6,
        vib_hidden: int = 256,
        op_hidden: int = 64,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.vibration_cnn = nn.Sequential(
            nn.Conv1d(vibration_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.vibration_lstm = nn.LSTM(
            input_size=64,
            hidden_size=vib_hidden,
            num_layers=2,
            batch_first=True,
        )

        self.operation_lstm = nn.LSTM(
            input_size=operation_features,
            hidden_size=op_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(vib_hidden + op_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_vibration: torch.Tensor, x_operation: torch.Tensor) -> torch.Tensor:
        # x_vibration: (batch, seq_len, 4, freq_bins)
        batch_size, seq_len, channels, freq_bins = x_vibration.shape

        # CNN은 timestep별 STFT 스펙트럼에서 주파수 패턴을 추출한다.
        vib = x_vibration.reshape(batch_size * seq_len, channels, freq_bins)
        vib = self.vibration_cnn(vib)

        # 다시 시퀀스로 복원한 뒤 LSTM으로 degradation 흐름을 학습한다.
        vib = vib.reshape(batch_size, seq_len, 64)
        _, (h_vib, _) = self.vibration_lstm(vib)
        h_vib_last = h_vib[-1]

        # 운전 데이터도 별도 LSTM으로 시간 흐름을 학습한다.
        _, (h_op, _) = self.operation_lstm(x_operation)
        h_op_last = h_op[-1]

        fused = torch.cat([h_vib_last, h_op_last], dim=1)
        return self.fusion(fused)


class AsymmetricRULLoss(nn.Module):
    """
    Er = 100 * (true_RUL - prediction) / true_RUL
    Er <= 0: exp(-ln(0.5) * Er / 20)
    Er > 0 : exp(+ln(0.5) * Er / 50)
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("ln_half", torch.tensor(math.log(0.5), dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(predictions)
        denominator = torch.clamp(torch.abs(targets), min=1e-6)
        er = 100.0 * (targets - predictions) / denominator
        exponent = torch.where(
            er <= 0,
            -self.ln_half * er / 20.0,
            self.ln_half * er / 50.0,
        )
        score = torch.exp(exponent)
        return 1.0 - score.mean()


def asymmetric_rul_score_np(predictions, targets) -> np.ndarray:
    """평가용 A_RUL score 계산."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    denominator = np.maximum(np.abs(targets), 1e-6)
    er = 100.0 * (targets - predictions) / denominator
    ln_half = np.log(0.5)
    exponent = np.where(
        er <= 0,
        -ln_half * er / 20.0,
        ln_half * er / 50.0,
    )
    return np.exp(exponent)


def create_model(vibration_channels: int = 4, operation_features: int = 6) -> STFTCNNLSTMRULModel:
    return STFTCNNLSTMRULModel(
        vibration_channels=vibration_channels,
        operation_features=operation_features,
    )
