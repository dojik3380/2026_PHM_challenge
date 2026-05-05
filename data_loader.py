"""STFT 기반 PHM RUL 예측용 데이터 로딩."""

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


def _time_from_tdms_index(index: int, max_time: float, total_files: int) -> float:
    if total_files <= 1:
        return max_time
    return max_time * index / float(total_files - 1)


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

    for index, tdms_path in enumerate(tdms_files):
        current_time = _time_from_tdms_index(index, max_time, total_files)
        channel_data = load_tdms_channels(tdms_path)
        vibration_steps.append(vibration_stft_timestep(channel_data))
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
