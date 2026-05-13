"""STFT 진동 특징과 운전 데이터 특징 생성."""

from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.signal import hilbert, stft
from scipy.stats import kurtosis, skew

from config import (
    OPERATION_FEATURES,
    SAMPLING_RATE,
    STFT_CACHE_DIR,
    STFT_CACHE_ENABLED,
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


def vibration_handcrafted_features(signal: np.ndarray) -> np.ndarray:
    """Handcrafted vibration features: RMS, Kurtosis, Crest Factor, Envelope, Band Energy."""
    if signal.size == 0:
        return np.zeros(5, dtype=np.float32)  # RMS, Kurtosis, Crest, Envelope_mean, Band_energy
    
    # RMS
    rms = np.sqrt(np.mean(signal ** 2))
    
    # Kurtosis
    kurt = kurtosis(signal, fisher=False, bias=False) if signal.size > 3 and np.std(signal) > 0 else 0.0
    
    # Crest Factor
    peak = np.max(np.abs(signal))
    crest = peak / rms if rms > 0 else 0.0
    
    # Envelope Feature (Hilbert transform)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope_mean = np.mean(envelope)
    
    # Frequency Band Energy (FFT 기반)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1/SAMPLING_RATE)
    # Band 1: 0-1kHz, Band 2: 1-5kHz, Band 3: 5-10kHz
    band1 = np.sum(np.abs(fft[(freqs >= 0) & (freqs < 1000)]) ** 2)
    band2 = np.sum(np.abs(fft[(freqs >= 1000) & (freqs < 5000)]) ** 2)
    band3 = np.sum(np.abs(fft[(freqs >= 5000) & (freqs < 10000)]) ** 2)
    total_energy = np.sum(np.abs(fft) ** 2)
    band_energy = (band1 + band2 + band3) / total_energy if total_energy > 0 else 0.0
    
    return np.array([rms, kurt, crest, envelope_mean, band_energy], dtype=np.float32)


def vibration_stft_timestep(
    tdms_file_path: str | Path,
) -> np.ndarray:
    """
    TDMS 파일 1개를 STFT timestep으로 변환 (캐시 지원).
    출력 shape: (4, freq_bins + handcrafted_features)
    """
    import hashlib
    import pickle
    from pathlib import Path
    
    tdms_path = Path(tdms_file_path)
    
    # 캐시 키 생성 (파일 경로 + 수정시간 기반)
    file_stat = tdms_path.stat()
    cache_key = hashlib.md5(f"{tdms_path}:{file_stat.st_mtime}".encode()).hexdigest()
    cache_file = STFT_CACHE_DIR / f"{cache_key}.pkl"
    
    # 캐시 히트 시 로드
    if STFT_CACHE_ENABLED and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # 캐시 로드 실패 시 재계산
    
    # 캐시 미스 시 계산
    from data_loader import load_tdms_channels
    channel_data = load_tdms_channels(tdms_path)
    
    normalized = {name.upper(): values for name, values in channel_data.items()}
    stft_vectors = []
    handcrafted_vectors = []
    
    for channel in VIBRATION_CHANNELS:
        signal = np.array(normalized.get(channel.upper(), []), dtype=np.float32)
        stft_vectors.append(stft_magnitude_vector(signal))
        handcrafted_vectors.append(vibration_handcrafted_features(signal))
    
    # STFT: (4, freq_bins), Handcrafted: (4, 5) -> concat to (4, freq_bins + 5)
    stft_stack = np.stack(stft_vectors, axis=0)
    handcrafted_stack = np.stack(handcrafted_vectors, axis=0)
    result = np.concatenate([stft_stack, handcrafted_stack], axis=1).astype(np.float32)
    
    # 캐시 저장
    if STFT_CACHE_ENABLED:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass  # 캐시 저장 실패 시 무시
    
    return result


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
        delta_torque = 0.0
        delta_temp_front = 0.0
        delta_temp_rear = 0.0
        torque_ma = base["torque"]
        temp_front_ma = base["temp_front"]
        temp_rear_ma = base["temp_rear"]
        prev = row  # dummy
    else:
        prev_idx = (operation_df["time_sec"] - previous_time).abs().idxmin()
        prev = operation_df.loc[prev_idx]
        dt = max(float(current_time - previous_time), 1e-6)
        front_slope = (base["temp_front"] - float(prev.get("temp_front", base["temp_front"]))) / dt
        rear_slope = (base["temp_rear"] - float(prev.get("temp_rear", base["temp_rear"]))) / dt
        
        # Delta features
        delta_torque = base["torque"] - float(prev.get("torque", base["torque"]))
        delta_temp_front = base["temp_front"] - float(prev.get("temp_front", base["temp_front"]))
        delta_temp_rear = base["temp_rear"] - float(prev.get("temp_rear", base["temp_rear"]))
        
        # Moving averages (recent 10 points)
        recent_df = operation_df[operation_df["time_sec"] <= current_time].tail(10)
        torque_ma = recent_df["torque"].mean() if not recent_df.empty else base["torque"]
        temp_front_ma = recent_df["temp_front"].mean() if not recent_df.empty else base["temp_front"]
        temp_rear_ma = recent_df["temp_rear"].mean() if not recent_df.empty else base["temp_rear"]

    return np.asarray(
        [
            base["torque"],
            base["speed"],
            base["temp_front"],
            base["temp_rear"],
            front_slope,
            rear_slope,
            delta_torque,
            delta_temp_front,
            delta_temp_rear,
            torque_ma,
            temp_front_ma,
            temp_rear_ma,
        ],
        dtype=np.float32,
    )


def augment_stft_features(stft_matrix: np.ndarray, aug_prob: float = 0.3) -> np.ndarray:
    """STFT 특징에 노이즈 억제 중심 증강 적용 (대회 특성 고려)"""
    augmented = stft_matrix.copy()
    
    # 1. 가벼운 가우시안 노이즈 (대회 노이즈 시뮬레이션)
    if np.random.random() < aug_prob * 0.5:  # 확률 낮춤
        noise_std = 0.02  # 기존 0.05 → 0.02로 감소
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented += noise
    
    # 2. 밝기 조정만 (콘트라스트 제외 - 노이즈 증가 방지)
    if np.random.random() < aug_prob * 0.7:
        brightness = np.random.uniform(0.95, 1.05)  # 범위 좁힘
        augmented *= brightness
    
    # 3. SpecAugment-style 마스킹 (노이즈 패턴 학습용)
    if np.random.random() < aug_prob * 0.4:
        freq_bins, time_steps = augmented.shape
        
        # 빈도 마스킹 (좁은 범위)
        if freq_bins > 1:
            max_freq_width = max(1, freq_bins // 8)
            mask_freq_width = np.random.randint(1, min(freq_bins, max_freq_width) + 1)
            mask_freq_start = np.random.randint(0, freq_bins - mask_freq_width + 1)
            augmented[mask_freq_start:mask_freq_start + mask_freq_width, :] *= 0.1  # 완전 마스킹 X
        
        # 시간 마스킹 (좁은 범위)
        if time_steps > 1:
            max_time_width = max(1, time_steps // 8)
            mask_time_width = np.random.randint(1, min(time_steps, max_time_width) + 1)
            mask_time_start = np.random.randint(0, time_steps - mask_time_width + 1)
            augmented[:, mask_time_start:mask_time_start + mask_time_width] *= 0.1
    
    return np.maximum(augmented, 0)  # 음수 방지


def augment_sequence_level(X_vib: np.ndarray, X_op: np.ndarray, y: float, aug_prob: float = 0.3) -> list:
    """시퀀스 레벨 증강 (시간축 조작)"""
    augmented = [(X_vib, X_op, y)]  # 원본 유지
    
    # 1. 가벼운 시간 반전 (물리적 의미 유지)
    if np.random.random() < aug_prob * 0.6:
        vib_reversed = np.flip(X_vib, axis=1)
        op_reversed = np.flip(X_op, axis=1)
        augmented.append((vib_reversed, op_reversed, y))
    
    # 2. 가벼운 시간 이동 (shift) 증강
    if np.random.random() < aug_prob * 0.4:
        shift = np.random.randint(-4, 5)
        if shift != 0:
            vib_shifted = np.roll(X_vib, shift, axis=0)
            op_shifted = np.roll(X_op, shift, axis=0)
            if shift > 0:
                vib_shifted[:shift] = X_vib[0:1]
                op_shifted[:shift] = X_op[0:1]
            else:
                vib_shifted[shift:] = X_vib[-1:]
                op_shifted[shift:] = X_op[-1:]
            augmented.append((vib_shifted, op_shifted, y))
    
    return augmented
