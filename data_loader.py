"""STFT 기반 PHM RUL 예측용 데이터 로딩."""

import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from nptdms import TdmsFile

from config import OPERATION_FEATURES, TRAIN_DIR
from features import operation_vector, vibration_stft_timestep


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")


def load_tdms_file(file_path):
    """작은 TDMS 테스트용 기본 로딩. 대용량 파일에는 사용하지 않는다."""
    tdms_file = TdmsFile.read(file_path)
    df = tdms_file.as_dataframe()
    return df


def load_tdms_channels(file_path):
    """대용량 대응 TDMS 로딩. 파일을 dataframe으로 만들지 않고 채널만 읽는다."""
    tdms_file = TdmsFile.read(file_path)
    data = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            data[channel.name] = channel[:]
    return data


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or ValueError(f"Could not read CSV: {path}")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(candidate.lower() in col_lower for candidate in candidates):
            return col
    return None


def load_operation_csv(csv_path: Path) -> pd.DataFrame:
    """운전 CSV를 읽고 컬럼명을 표준화한다."""
    df = _read_csv(csv_path)
    mapping = {
        _find_column(df.columns, ("time_sec", "time", "sec")): "time_sec",
        _find_column(df.columns, ("torque",)): "torque",
        _find_column(df.columns, ("speed", "rpm")): "speed",
        _find_column(df.columns, ("temp_front", "front")): "temp_front",
        _find_column(df.columns, ("temp_rear", "rear")): "temp_rear",
    }
    mapping = {old: new for old, new in mapping.items() if old is not None}
    df = df.rename(columns=mapping)

    if "time_sec" not in df.columns:
        raise ValueError(f"{csv_path} must contain time_sec or an equivalent time column.")

    for column in ("time_sec", "torque", "speed", "temp_front", "temp_rear"):
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df.sort_values("time_sec").reset_index(drop=True)


def compute_rul(operation_df: pd.DataFrame, current_time: float) -> float:
    """RUL = max_time - current_time."""
    return float(operation_df["time_sec"].max() - current_time)


def _case_name_from_operation(csv_path: Path) -> str:
    return csv_path.stem.replace("_Operation", "")


def _find_vibration_dir(case_name: str, root_dir: Path) -> Optional[Path]:
    candidates = [
        root_dir / f"{case_name}_Vibration" / f"{case_name}_Vibration",
        root_dir / f"{case_name}_Vibration",
        root_dir / case_name / case_name,
        root_dir / case_name,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.tdms")):
            return candidate
    for candidate in sorted(root_dir.rglob(f"{case_name}*")):
        if candidate.is_dir() and any(candidate.glob("*.tdms")):
            return candidate
    return None


def discover_cases(root_dir: Path) -> List[Tuple[str, Path, Path]]:
    """운전 CSV와 TDMS 진동 폴더가 모두 있는 case를 찾는다."""
    root_dir = Path(root_dir)
    cases = []
    for csv_path in sorted(root_dir.glob("*_Operation.csv")):
        case_name = _case_name_from_operation(csv_path)
        vibration_dir = _find_vibration_dir(case_name, root_dir)
        if vibration_dir is not None:
            cases.append((case_name, csv_path, vibration_dir))
    return cases


def discover_vibration_cases(root_dir: Path) -> List[Tuple[str, Path]]:
    """Find validation/test cases that contain TDMS files without requiring operation CSVs."""
    root_dir = Path(root_dir)
    cases = []

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        vibration_dir = _find_vibration_dir(child.name, root_dir)
        if vibration_dir is not None:
            cases.append((child.name, vibration_dir))

    if cases:
        order = sort_cases_for_submission([name for name, _ in cases])
        name_to_dir = dict(cases)
        return [(name, name_to_dir[name]) for name in order]

    if any(root_dir.glob("*.tdms")):
        return [(root_dir.name, root_dir)]

    return []


def _parse_tdms_file_index(tdms_path: Path) -> Optional[int]:
    """TDMS 파일명에서 숫자 인덱스 추출 (예: 000126.tdms → 126)."""
    stem = tdms_path.stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else None


def _detect_tdms_index_gaps(tdms_files: List[Path]) -> List[int]:
    """연속되지 않는 인덱스(누락 파일) 목록."""
    indices = [_parse_tdms_file_index(p) for p in tdms_files]
    if not indices or any(i is None for i in indices):
        return []
    sorted_idx = sorted(set(indices))
    gaps = []
    for expected in range(sorted_idx[0], sorted_idx[-1] + 1):
        if expected not in sorted_idx:
            gaps.append(expected)
    return gaps


def _time_from_tdms_index(index: int, max_time: float, total_files: int) -> float:
    if total_files <= 1:
        return max_time
    return max_time * index / float(total_files - 1)


def _align_tdms_times(
    tdms_files: List[Path],
    operation_df: pd.DataFrame,
    case_name: str,
) -> List[float]:
    """
    TDMS 파일별 time_sec 정렬.

    1) 파일명 숫자 인덱스 → [0, max_time] 선형 매핑 (누락 구간은 실제 파일만 사용)
    2) train4 등 공지된 누락 케이스: 인덱스 갭 경고
    3) 폴백: 파일 순서 기반 선형 매핑
    4) 보조: 10분(600s) 간격으로 CSV time_sec 보간
    """
    max_time = float(operation_df["time_sec"].max())
    n = len(tdms_files)
    if n == 0:
        return []

    indices = [_parse_tdms_file_index(p) for p in tdms_files]
    gaps = _detect_tdms_index_gaps(tdms_files)
    if gaps and case_name in CASES_WITH_KNOWN_MISSING_TDMS:
        warnings.warn(
            f"[{case_name}] TDMS index gap(s) detected (known dataset issue): missing {gaps[-3:]}{'...' if len(gaps) > 3 else ''}",
            stacklevel=2,
        )
    elif gaps:
        warnings.warn(
            f"[{case_name}] TDMS index gap(s): {gaps[-3:]}{'...' if len(gaps) > 3 else ''}",
            stacklevel=2,
        )

    if indices and all(i is not None for i in indices):
        min_idx, max_idx = min(indices), max(indices)
        if max_idx > min_idx:
            times = [max_time * (idx - min_idx) / float(max_idx - min_idx) for idx in indices]
            return times

    # 순서 기반 선형 매핑
    times = [_time_from_tdms_index(i, max_time, n) for i in range(n)]

    # CSV time_sec과 600s 간격 앵커로 미세 보정 (가능한 경우)
    if max_time > 0 and n > 1:
        csv_times = operation_df["time_sec"].to_numpy(dtype=np.float64)
        for i, t in enumerate(times):
            anchor = min(i * TDMS_ACQUISITION_INTERVAL_SEC, max_time)
            nearest = csv_times[np.argmin(np.abs(csv_times - anchor))]
            times[i] = float(0.7 * t + 0.3 * nearest)

    return times


def _case_timesteps(
    operation_csv: Path,
    vibration_dir: Path,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TDMS 파일별 timestep 생성.
    진동 shape: (num_steps, 4, freq_bins), 운전 shape: (num_steps, 6)
    """
    operation_df = load_operation_csv(operation_csv)
    max_time = float(operation_df["time_sec"].max())
    all_tdms_files = sorted(Path(vibration_dir).glob("*.tdms"))
    if not all_tdms_files:
        raise ValueError(f"No TDMS files found in {vibration_dir}")

    total_files = len(all_tdms_files)
    tdms_files = all_tdms_files[:max_files] if max_files is not None else all_tdms_files

    vibration_steps = []
    operation_steps = []
    times = []
    previous_time = None

    case_name = _case_name_from_operation(operation_csv)
    aligned_times = _align_tdms_times(tdms_files, operation_df, case_name)

    for index, tdms_path in enumerate(tdms_files):
        current_time = aligned_times[index]
        vibration_steps.append(vibration_stft_timestep(tdms_path))
        operation_steps.append(operation_vector(operation_df, current_time, previous_time))
        times.append(current_time)
        previous_time = current_time

    return (
        np.asarray(vibration_steps, dtype=np.float32),
        np.asarray(operation_steps, dtype=np.float32),
        np.asarray(times, dtype=np.float32),
    )


def build_case_sequences(
    operation_csv: Path,
    vibration_dir: Path,
    case_name: str,
    window_size: int,
    stride: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """정렬된 진동/운전 timestep에 sliding window를 적용한다."""
    operation_df = load_operation_csv(operation_csv)
    max_files = None
    if max_samples is not None:
        max_files = window_size + max(0, max_samples - 1) * stride

    vibration_steps, operation_steps, times = _case_timesteps(
        operation_csv,
        vibration_dir,
        max_files=max_files,
    )

    X_vibration = []
    X_operation = []
    y = []
    metadata = []

    for start in range(0, len(times) - window_size + 1, stride):
        if max_samples is not None and len(y) >= max_samples:
            break
        end = start + window_size
        current_time = float(times[end - 1])
        X_vibration.append(vibration_steps[start:end])
        X_operation.append(operation_steps[start:end])
        y.append(compute_rul(operation_df, current_time))
        metadata.append({"case_name": case_name, "start_time": float(times[start]), "time_sec": current_time})

    if not y:
        raise ValueError(f"Window size {window_size} is larger than available TDMS files in {vibration_dir}")

    return (
        np.asarray(X_vibration, dtype=np.float32),
        np.asarray(X_operation, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def build_inference_sequence(
    vibration_dir: Path,
    case_name: str,
    window_size: int,
    max_samples: Optional[int] = None,
    use_vibration_proxy: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build one validation/test sequence from the latest available TDMS window."""
    if use_vibration_proxy is None:
        use_vibration_proxy = USE_VIBRATION_PROXY_AT_INFERENCE

    tdms_files = sorted(Path(vibration_dir).glob("*.tdms"))
    if max_samples is not None:
        tdms_files = tdms_files[:max_samples]
    if len(tdms_files) < window_size:
        raise ValueError(f"Window size {window_size} is larger than available TDMS files in {vibration_dir}")

    selected_files = tdms_files[-window_size:]
    vibration_steps = []
    for tdms_path in selected_files:
        vibration_steps.append(vibration_stft_timestep(tdms_path))

    vib_window = np.asarray(vibration_steps, dtype=np.float32)
    if use_vibration_proxy:
        op_window = vibration_proxy_operation_sequence(
            vib_window,
            interval_sec=TDMS_ACQUISITION_INTERVAL_SEC,
        )
    else:
        op_window = np.zeros((window_size, len(OPERATION_FEATURES)), dtype=np.float32)

    X_vibration = np.asarray([vib_window], dtype=np.float32)
    X_operation = np.asarray([op_window], dtype=np.float32)
    metadata = pd.DataFrame(
        [
            {
                "case_name": case_name,
                "start_file": selected_files[0].stem,
                "end_file": selected_files[-1].stem,
                "num_files": len(tdms_files),
                "inference_operation": "vibration_proxy" if use_vibration_proxy else "zeros",
            }
        ]
    )
    return X_vibration, X_operation, metadata


def apply_data_augmentation(X_vib_batch: np.ndarray, X_op_batch: np.ndarray, y_batch: np.ndarray, 
                          aug_prob: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """통합 데이터 증강 적용 (훈련 시에만 사용)"""
    from features import augment_stft_features, augment_sequence_level
    
    vib_augmented = []
    op_augmented = []
    y_augmented = []
    
    for vib_seq, op_seq, y in zip(X_vib_batch, X_op_batch, y_batch):
        # 원본 추가
        vib_augmented.append(vib_seq)
        op_augmented.append(op_seq)
        y_augmented.append(y)
        
        # STFT 특징 증강 (각 채널별로)
        vib_stft_aug = vib_seq.copy()
        for ch in range(vib_seq.shape[1]):  # 채널별
            for t in range(vib_seq.shape[0]):  # 타임스텝별
                stft_features = vib_seq[t, ch]  # (freq_bins,) 
                vib_stft_aug[t, ch] = augment_stft_features(
                    stft_features.reshape(-1, 1), aug_prob
                ).flatten()
        
        vib_augmented.append(vib_stft_aug)
        op_augmented.append(op_seq)  # operation은 그대로
        y_augmented.append(y)
        
        # 시퀀스 레벨 증강
        seq_augmented = augment_sequence_level(vib_seq, op_seq, y, aug_prob)
        for vib_aug, op_aug, y_aug in seq_augmented[1:]:  # 원본 제외
            vib_augmented.append(vib_aug)
            op_augmented.append(op_aug)
            y_augmented.append(y_aug)
    
    return (
        np.array(vib_augmented, dtype=np.float32),
        np.array(op_augmented, dtype=np.float32), 
        np.array(y_augmented, dtype=np.float32)
    )


def load_inference_dataset(
    root_dir: Path,
    window_size: int = 32,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load validation/test TDMS-only cases for RUL submission inference."""
    vib_batches = []
    op_batches = []
    metadata_frames = []

    for case_name, vibration_dir in discover_vibration_cases(root_dir):
        X_vib, X_op, metadata = build_inference_sequence(
            vibration_dir=vibration_dir,
            case_name=case_name,
            window_size=window_size,
            max_samples=max_samples,
        )
        vib_batches.append(X_vib)
        op_batches.append(X_op)
        metadata_frames.append(metadata)

    if not vib_batches:
        raise ValueError(f"No TDMS validation/test cases found in {root_dir}")

    return (
        np.concatenate(vib_batches, axis=0),
        np.concatenate(op_batches, axis=0),
        pd.concat(metadata_frames, ignore_index=True),
    )


def load_dataset(
    root_dir: Path = TRAIN_DIR,
    window_size: int = 32,
    stride: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    전체 case를 로딩한다.
    X_vibration: (batch, seq_len, 4, freq_bins)
    X_operation: (batch, seq_len, features)
    y: (batch,)
    """
    vib_batches = []
    op_batches = []
    targets = []
    metadata_frames = []

    for case_name, operation_csv, vibration_dir in discover_cases(root_dir):
        loaded = sum(len(batch) for batch in targets)
        remaining = None if max_samples is None else max_samples - loaded
        if remaining is not None and remaining <= 0:
            break

        X_vib, X_op, y, metadata = build_case_sequences(
            operation_csv=operation_csv,
            vibration_dir=vibration_dir,
            case_name=case_name,
            window_size=window_size,
            stride=stride,
            max_samples=remaining,
        )
        vib_batches.append(X_vib)
        op_batches.append(X_op)
        targets.append(y)
        metadata_frames.append(metadata)

    if not vib_batches:
        raise ValueError(f"No cases with operation CSV + TDMS files found in {root_dir}")

    return (
        np.concatenate(vib_batches, axis=0),
        np.concatenate(op_batches, axis=0),
        np.concatenate(targets, axis=0),
        pd.concat(metadata_frames, ignore_index=True),
    )


def operation_feature_names() -> tuple[str, ...]:
    return OPERATION_FEATURES
