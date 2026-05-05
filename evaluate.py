"""STFT + CNN + LSTM PHM RUL 모델 평가."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DEVICE, MODEL_PATH, PREDICTION_PATH, STRIDE, TEST_DIR, WINDOW_SIZE
from data_loader import load_dataset
from model import asymmetric_rul_score_np, create_model


def _apply_standardization(array: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """학습 때 저장한 평균/표준편차로 표준화한다."""
    return ((array - mean.cpu().numpy()) / std.cpu().numpy()).astype(np.float32)


def evaluate_model(
    data_dir: Path = TEST_DIR,
    model_path: Path = MODEL_PATH,
    output_path: Optional[Path] = PREDICTION_PATH,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    max_samples: Optional[int] = None,
) -> dict:
    """추론 후 MAE, RMSE, A_RUL을 계산한다."""
    checkpoint = torch.load(model_path, map_location="cpu")
    X_vib, X_op, y, metadata = load_dataset(
        root_dir=data_dir,
        window_size=window_size,
        stride=stride,
        max_samples=max_samples,
    )

    X_vib = _apply_standardization(X_vib, checkpoint["vibration_mean"], checkpoint["vibration_std"])
    X_op = _apply_standardization(X_op, checkpoint["operation_mean"], checkpoint["operation_std"])

    device = torch.device(DEVICE)
    model = create_model(
        vibration_channels=checkpoint["vibration_channels"],
        operation_features=checkpoint["operation_features"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(y), 512):
            vib_batch = torch.from_numpy(X_vib[start:start + 512]).to(device)
            op_batch = torch.from_numpy(X_op[start:start + 512]).to(device)
            predictions.append(model(vib_batch, op_batch).cpu().numpy().reshape(-1))
    y_pred = np.concatenate(predictions)

    arul = asymmetric_rul_score_np(y_pred, y)
    metrics = {
        "MAE": float(mean_absolute_error(y, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
        "A_RUL": float(np.mean(arul)),
    }

    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"A_RUL: {metrics['A_RUL']:.6f}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = metadata.copy()
        results["true_RUL"] = y
        results["predicted_RUL"] = y_pred
        results["error"] = y_pred - y
        results["A_RUL"] = arul
        results.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
