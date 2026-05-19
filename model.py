"""STFT + CNN + LSTM 멀티브랜치 RUL 모델."""

import math

import numpy as np
import torch
import torch.nn as nn

from config import (
    ASYMMETRIC_WEIGHT,
    DROPOUT,
    HUBER_WEIGHT,
    OPERATION_FEATURES,
    OVER_EST_PENALTY_SCALE,
    UNDER_EST_PENALTY_SCALE,
    VIBRATION_FEATURES_PER_CHANNEL,
)

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D CNN.
    특정 주파수 채널(Fault Harmonic)에 어텐션을 주어 노이즈를 억제합니다.
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



class STFTCNNLSTMRULModel(nn.Module):
    """
    진동 branch: STFT 입력 -> CNN -> LSTM
    운전 branch: 운전 시퀀스 -> LSTM
    Fusion head: 두 branch 출력을 결합해 RUL 예측
    """

    def __init__(
        self,
        vibration_channels: int = 4,
        operation_features: int = len(OPERATION_FEATURES),
        vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
        vib_hidden: int = 128,
        op_hidden: int = 32,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.vibration_cnn = nn.Sequential(
            nn.Conv1d(vibration_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock1D(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock1D(64),
            nn.Flatten(),
        )
        
        # Calculate flattened size dynamically
        dummy_input = torch.zeros(1, vibration_channels, vibration_features)
        with torch.no_grad():
            cnn_out_size = self.vibration_cnn(dummy_input).shape[1]
            
        self.vibration_projection = nn.Sequential(
            nn.Linear(cnn_out_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.vibration_lstm = nn.LSTM(
            input_size=128,
            hidden_size=vib_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.operation_lstm = nn.LSTM(
            input_size=operation_features,
            hidden_size=op_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(vib_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.last_attn_weights = None

        self.fusion = nn.Sequential(
            nn.Linear(vib_hidden * 2 + op_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_vibration: torch.Tensor, x_operation: torch.Tensor) -> torch.Tensor:
        # Modality Dropout: 추론(Test) 시 운전 데이터가 0으로 들어오는 환경에 대비하기 위해
        # 훈련(Train) 시 50% 확률로 운전 데이터를 모두 0으로 지워버려 진동 데이터에 대한 의존도를 높임
        if self.training and torch.rand(1).item() < 0.5:
            x_operation = torch.zeros_like(x_operation)
            
        # x_vibration: (batch, seq_len, 4, freq_bins)
        batch_size, seq_len, channels, freq_bins = x_vibration.shape

        # CNN은 timestep별 STFT 스펙트럼에서 주파수 패턴을 추출한다.
        vib = x_vibration.reshape(batch_size * seq_len, channels, freq_bins)
        vib = self.vibration_cnn(vib)
        vib = self.vibration_projection(vib)

        # 다시 시퀀스로 복원한 뒤 LSTM으로 degradation 흐름을 학습한다.
        vib = vib.reshape(batch_size, seq_len, 128)
        vib_out, _ = self.vibration_lstm(vib)
        
        # Temporal Attention Pooling
        attn_scores = self.temporal_attention(vib_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # GPU memory leak 방지를 위해 detach.cpu()로 저장하여 시각화 모듈에서 꺼내 쓸 수 있도록 함
        self.last_attn_weights = attn_weights.detach().cpu()
        
        h_vib_attended = torch.sum(vib_out * attn_weights, dim=1)

        # 운전 데이터도 별도 BiLSTM으로 시간 흐름을 학습한다.
        _, (h_op, _) = self.operation_lstm(x_operation)
        h_op_last = torch.cat([h_op[-2], h_op[-1]], dim=1)  # bidirectional concat

        # Attention fusion
        fused_input = torch.cat([h_vib_attended, h_op_last], dim=1)

        return self.fusion(fused_input)


class AsymmetricRULLoss(nn.Module):
    """
    Asymmetric penalty for RUL prediction.
    Overestimation (prediction > target) gets higher penalty.
    """

    def __init__(self, over_scale: float = OVER_EST_PENALTY_SCALE, under_scale: float = UNDER_EST_PENALTY_SCALE):
        super().__init__()
        self.over_scale = over_scale
        self.under_scale = under_scale

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(predictions)
        error = predictions - targets  # positive: overestimation
        
        # Asymmetric penalty
        penalty = torch.where(
            error > 0,
            torch.exp(error / self.over_scale) - 1,  # overestimation penalty
            torch.exp(-error / self.under_scale) - 1,  # underestimation penalty
        )
        return penalty.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of HuberLoss and AsymmetricRULLoss.
    """

    def __init__(self, huber_weight: float = HUBER_WEIGHT, asymmetric_weight: float = ASYMMETRIC_WEIGHT):
        super().__init__()
        self.huber = nn.HuberLoss()
        self.asymmetric = AsymmetricRULLoss()
        self.huber_weight = huber_weight
        self.asymmetric_weight = asymmetric_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(predictions, targets)
        asymmetric_loss = self.asymmetric(predictions, targets)
        return self.huber_weight * huber_loss + self.asymmetric_weight * asymmetric_loss  #로스 함수 


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


def create_model(vibration_channels: int = 4, operation_features: int = len(OPERATION_FEATURES), vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL) -> STFTCNNLSTMRULModel:
    return STFTCNNLSTMRULModel(
        vibration_channels=vibration_channels,
        operation_features=operation_features,
        vibration_features=vibration_features,
        vib_hidden=128,
        op_hidden=32,
    )
