"""STFT 진동 특징과 운전 데이터 특징 생성."""

from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.stats import kurtosis, skew

from config import (
    OPERATION_FEATURES,
    SAMPLING_RATE,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    VIBRATION_CHANNELS,
)


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def vibration_statistics(signal: Iterable[float]) -> Dict[str, float]:
    """필요 시 사용할 수 있는 시간영역 통계 특징."""
    arr = _finite_array(signal)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "rms": 0.0, "kurtosis": 0.0, "skewness": 0.0}

    std = float(np.std(arr))
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "kurtosis": float(kurtosis(arr, fisher=False, bias=False)) if arr.size > 3 and std > 0 else 0.0,
        "skewness": float(skew(arr, bias=False)) if arr.size > 2 and std > 0 else 0.0,
    }


def stft_magnitude_vector(signal: Iterable[float]) -> np.ndarray:
    """
    채널 1개의 raw 진동 신호를 STFT magnitude 벡터로 변환한다.
    STFT time 축은 평균내서 freq_bins 길이의 벡터로 만든다.
    """
    arr = _finite_array(signal)
    if arr.size == 0:
        return np.zeros(STFT_FREQ_BINS, dtype=np.float32)

    if arr.size < STFT_NPERSEG:
        arr = np.pad(arr, (0, STFT_NPERSEG - arr.size))

    _, _, zxx = stft(
        arr,
        fs=SAMPLING_RATE,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(zxx)
    freq_vector = np.log1p(magnitude).mean(axis=1)

    if freq_vector.size < STFT_FREQ_BINS:
        freq_vector = np.pad(freq_vector, (0, STFT_FREQ_BINS - freq_vector.size))
    elif freq_vector.size > STFT_FREQ_BINS:
        freq_vector = freq_vector[:STFT_FREQ_BINS]

    return freq_vector.astype(np.float32)


def vibration_stft_timestep(
    channel_data: Mapping[str, Iterable[float]],
) -> np.ndarray:
    """
    TDMS 파일 1개를 하나의 STFT timestep으로 변환한다.
    출력 shape: (4, freq_bins)
    """
    normalized = {name.upper(): values for name, values in channel_data.items()}
    channel_vectors = []
    for channel in VIBRATION_CHANNELS:
        channel_vectors.append(stft_magnitude_vector(normalized.get(channel.upper(), [])))
    return np.stack(channel_vectors, axis=0).astype(np.float32)


def operation_vector(operation_df: pd.DataFrame, current_time: float, previous_time: float | None) -> np.ndarray:
    """
    진동 timestep 시간에 가장 가까운 운전 데이터를 맞춘다.
    온도 slope는 이전 timestep 대비 변화량 / 시간 변화량이다.
    """
    if operation_df.empty:
        return np.zeros(len(OPERATION_FEATURES), dtype=np.float32)

    nearest_idx = (operation_df["time_sec"] - current_time).abs().idxmin()
    row = operation_df.loc[nearest_idx]
    base = {
        "torque": float(row.get("torque", 0.0)),
        "speed": float(row.get("speed", 0.0)),
        "temp_front": float(row.get("temp_front", 0.0)),
        "temp_rear": float(row.get("temp_rear", 0.0)),
    }

    if previous_time is None or previous_time == current_time:
        front_slope = 0.0
        rear_slope = 0.0
    else:
        prev_idx = (operation_df["time_sec"] - previous_time).abs().idxmin()
        prev = operation_df.loc[prev_idx]
        dt = max(float(current_time - previous_time), 1e-6)
        front_slope = (base["temp_front"] - float(prev.get("temp_front", base["temp_front"]))) / dt
        rear_slope = (base["temp_rear"] - float(prev.get("temp_rear", base["temp_rear"]))) / dt

    return np.asarray(
        [
            base["torque"],
            base["speed"],
            base["temp_front"],
            base["temp_rear"],
            front_slope,
            rear_slope,
        ],
        dtype=np.float32,
    )
