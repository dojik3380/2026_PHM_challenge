"""PHM STFT + CNN + LSTM RUL 파이프라인 CLI.

예시:
    python main.py train
    python main.py evaluate
"""

import argparse
from pathlib import Path

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, PREDICTION_PATH, STRIDE, TEST_DIR, TRAIN_DIR, WINDOW_SIZE
from evaluate import evaluate_model
from train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHM STFT + CNN + LSTM RUL prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the STFT CNN+LSTM model")
    train_parser.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    train_parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    train_parser.add_argument("--stride", type=int, default=STRIDE)
    train_parser.add_argument("--max-samples", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model")
    eval_parser.add_argument("--data-dir", type=Path, default=TEST_DIR)
    eval_parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    eval_parser.add_argument("--output-path", type=Path, default=PREDICTION_PATH)
    eval_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    eval_parser.add_argument("--stride", type=int, default=STRIDE)
    eval_parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_model(
            data_dir=args.data_dir,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            window_size=args.window_size,
            stride=args.stride,
            max_samples=args.max_samples,
        )
    elif args.command == "evaluate":
        evaluate_model(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_path=args.output_path,
            window_size=args.window_size,
            stride=args.stride,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
