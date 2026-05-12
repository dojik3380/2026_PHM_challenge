"""STFT + CNN + LSTM PHM RUL 모델 평가."""

from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DEVICE, MODEL_PATH, PREDICTION_PATH, STRIDE, TEST_DIR, VALIDATION_PREDICTION_PATH, WINDOW_SIZE
from data_loader import discover_cases, load_dataset, load_inference_dataset
from model import asymmetric_rul_score_np, create_model


def _apply_standardization(array: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """학습 때 저장한 평균/표준편차로 표준화한다."""
    return ((array - mean.cpu().numpy()) / std.cpu().numpy()).astype(np.float32)


def _load_trained_model(checkpoint: dict) -> torch.nn.Module:
    device = torch.device(DEVICE)
    model = create_model(
        vibration_channels=checkpoint["vibration_channels"],
        operation_features=checkpoint["operation_features"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _has_operation_cases(root_dir: Path) -> bool:
    """Return True when the directory contains labeled operation cases."""
    return bool(discover_cases(root_dir))


def _save_prediction_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".xlsx", ".xls"):
        _save_xlsx(df, output_path)
    else:
        df.to_csv(output_path, index=False)


def _predict(
    model: torch.nn.Module,
    X_vib: np.ndarray,
    X_op: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    device = torch.device(DEVICE)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X_vib), batch_size):
            vib_batch = torch.from_numpy(X_vib[start:start + batch_size]).to(device)
            op_batch = torch.from_numpy(X_op[start:start + batch_size]).to(device)
            predictions.append(model(vib_batch, op_batch).cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def _write_minimal_xlsx(df: pd.DataFrame, output_path: Path) -> None:
    """openpyxl 없이 File/RUL_Score 형식의 최소 xlsx 파일을 저장한다."""
    rows = [list(df.columns)] + df.astype(object).values.tolist()

    def cell_ref(row_idx: int, col_idx: int) -> str:
        return f"{chr(ord('A') + col_idx)}{row_idx}"

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = cell_ref(row_idx, col_idx)
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )

    with ZipFile(output_path, "w", ZIP_DEFLATED) as xlsx:
        xlsx.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>',
        )
        xlsx.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        xlsx.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>',
        )
        xlsx.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>',
        )
        xlsx.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _save_xlsx(df: pd.DataFrame, output_path: Path) -> None:
    """openpyxl이 있으면 pandas를 쓰고, 없으면 내장 writer를 쓴다."""
    try:
        df.to_excel(output_path, index=False)
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        _write_minimal_xlsx(df, output_path)


def evaluate_model(
    data_dir: Path = TEST_DIR,
    model_path: Path = MODEL_PATH,
    output_path: Optional[Path] = PREDICTION_PATH,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    max_samples: Optional[int] = None,
) -> dict:
    """추론 후 MAE, RMSE, A_RUL을 계산한다.

    TDMS-only 테스트 세트가 주어지면 `predict_validation` 방식으로 파일별 RUL 스코어만 생성합니다.
    """
    checkpoint = torch.load(model_path, map_location="cpu")

    if _has_operation_cases(data_dir):
        X_vib, X_op, y, metadata = load_dataset(
            root_dir=data_dir,
            window_size=window_size,
            stride=stride,
            max_samples=max_samples,
        )

        X_vib = _apply_standardization(X_vib, checkpoint["vibration_mean"], checkpoint["vibration_std"])
        X_op = _apply_standardization(X_op, checkpoint["operation_mean"], checkpoint["operation_std"])

        model = _load_trained_model(checkpoint)
        y_pred = _predict(model, X_vib, X_op)
        y_pred = np.expm1(y_pred)
        print(f"y_pred[:5] after expm1: {y_pred[:5]}")

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

    print(f"No labeled operation cases found in {data_dir}. Running TDMS-only inference instead.")
    validation_results = predict_validation(
        data_dir=data_dir,
        model_path=model_path,
        output_path=output_path if output_path is not None else VALIDATION_PREDICTION_PATH,
        window_size=window_size,
        max_samples=max_samples,
    )

    return {
        "mode": "tdms_only_inference",
        "cases": len(validation_results),
        "output_path": str(output_path if output_path is not None else VALIDATION_PREDICTION_PATH),
    }


def predict_validation(
    data_dir: Path = TEST_DIR,
    model_path: Path = MODEL_PATH,
    output_path: Optional[Path] = VALIDATION_PREDICTION_PATH,
    window_size: int = WINDOW_SIZE,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Create a validation RUL score Excel file from TDMS-only cases."""
    checkpoint = torch.load(model_path, map_location="cpu")
    X_vib, X_op, metadata = load_inference_dataset(
        root_dir=data_dir,
        window_size=window_size,
        max_samples=max_samples,
    )

    X_vib = _apply_standardization(X_vib, checkpoint["vibration_mean"], checkpoint["vibration_std"])
    X_op = _apply_standardization(X_op, checkpoint["operation_mean"], checkpoint["operation_std"])

    model = _load_trained_model(checkpoint)
    y_pred = _predict(model, X_vib, X_op)
    
    # RUL log scaling 복원
    y_pred = np.expm1(y_pred)

    file_names = metadata["case_name"].astype(str).tolist()
    if not all(name.lower().startswith("validation") for name in file_names):
        file_names = [f"Validation{i + 1}" for i in range(len(file_names))]

    results = pd.DataFrame(
        {
            "File": file_names,
            "RUL_Score": np.rint(np.maximum(y_pred, 0.0)).astype(int),
        }
    )

    if output_path is not None:
        _save_prediction_output(results, Path(output_path))
        print(f"Saved validation RUL score file to {output_path}")

    return results


if __name__ == "__main__":
    evaluate_model()
