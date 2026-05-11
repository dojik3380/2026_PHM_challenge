"""STFT + CNN + LSTM PHM RUL 모델 학습."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    RANDOM_STATE,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    STRIDE,
    TEST_SIZE,
    TRAIN_DIR,
    WINDOW_SIZE,
)
from data_loader import load_dataset, operation_feature_names
from model import AsymmetricRULLoss, CombinedLoss, create_model


def _standardize_temporal_array(
    train_array: np.ndarray,
    val_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    batch와 sequence 축을 합쳐 표준화한다.
    진동은 (4, freq_bins), 운전은 (features,) 단위로 평균/표준편차를 가진다.
    """
    feature_shape = train_array.shape[2:]
    train_flat = train_array.reshape(-1, *feature_shape)
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    train_scaled = ((train_array - mean) / std).astype(np.float32)
    val_scaled = ((val_array - mean) / std).astype(np.float32)
    return (
        train_scaled,
        val_scaled,
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def train_model(
    data_dir: Path = TRAIN_DIR,
    model_path: Path = MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    random_state: int = RANDOM_STATE,
    max_samples: Optional[int] = None,
) -> Path:
    """Case-level split, DataLoader, AdamW, 비대칭 RUL loss로 학습한다."""
    X_vib, X_op, y, metadata = load_dataset(
        root_dir=data_dir,
        window_size=window_size,
        stride=stride,
        max_samples=max_samples,
    )
    if len(y) < 2:
        raise ValueError("At least two sequence samples are required for train/validation split.")

    # Case-level split: metadata['case_name']을 group으로 사용
    groups = metadata['case_name'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
    train_idx, val_idx = next(gss.split(X=np.arange(len(y)), y=None, groups=groups))

    X_vib_train, X_vib_val, vib_mean, vib_std = _standardize_temporal_array(X_vib[train_idx], X_vib[val_idx])
    X_op_train, X_op_val, op_mean, op_std = _standardize_temporal_array(X_op[train_idx], X_op[val_idx])
    y_train = y[train_idx].astype(np.float32)
    y_val = y[val_idx].astype(np.float32)

    # RUL log scaling 적용
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_vib_train),
            torch.from_numpy(X_op_train),
            torch.from_numpy(y_train).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_vib_val),
            torch.from_numpy(X_op_val),
            torch.from_numpy(y_val).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device(DEVICE)
    model = create_model(
        vibration_channels=X_vib.shape[2],
        operation_features=X_op.shape[-1],
    ).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"Loaded {len(y)} sequences from {data_dir}")
    print(f"X_vibration: {X_vib.shape} | X_operation: {X_op.shape} | y: {y.shape}")
    print(f"STFT: nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}, freq_bins={STFT_FREQ_BINS}")
    print(f"Device: {device} | Train: {len(train_idx)} | Val: {len(val_idx)}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_vib, batch_op, batch_y in train_loader:
            batch_vib = batch_vib.to(device)
            batch_op = batch_op.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_vib, batch_op)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch_y.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_vib, batch_op, batch_y in val_loader:
                batch_vib = batch_vib.to(device)
                batch_op = batch_op.to(device)
                batch_y = batch_y.to(device)
                val_loss += criterion(model(batch_vib, batch_op), batch_y).item() * batch_y.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "window_size": window_size,
            "stride": stride,
            "stft_nperseg": STFT_NPERSEG,
            "stft_noverlap": STFT_NOVERLAP,
            "stft_freq_bins": STFT_FREQ_BINS,
            "vibration_channels": X_vib.shape[2],
            "operation_features": X_op.shape[-1],
            "operation_feature_names": operation_feature_names(),
            "vibration_mean": vib_mean,
            "vibration_std": vib_std,
            "operation_mean": op_mean,
            "operation_std": op_std,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")
    return model_path


if __name__ == "__main__":
    train_model()
