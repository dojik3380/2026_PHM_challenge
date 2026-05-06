"""PHM RUL 예측 파이프라인 설정."""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
VALIDATION_DIR = DATA_DIR / "Validation"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "stft_cnn_lstm_rul.pt"
PREDICTION_PATH = RESULTS_DIR / "evaluation_predictions.csv"
TEAM_NAME = "team_name"
VALIDATION_PREDICTION_PATH = RESULTS_DIR / f"{TEAM_NAME}_validation.xlsx"

VIBRATION_CHANNELS = ("CH1", "CH2", "CH3", "CH4")
OPERATION_FEATURES = (
    "torque",
    "speed",
    "temp_front",
    "temp_rear",
    "temp_front_slope",
    "temp_rear_slope",
)

# STFT 설정: 25.6 kHz 진동 신호를 주파수 영역으로 변환한다.
SAMPLING_RATE = 25_600
STFT_NPERSEG = 256
STFT_NOVERLAP = STFT_NPERSEG // 2
STFT_FREQ_BINS = STFT_NPERSEG // 2 + 1

WINDOW_SIZE = 32
STRIDE = 4
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
