import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_monotonicity(rul_sequence: np.ndarray) -> float:
    """단조 감소성을 수치화합니다."""
    diffs = np.diff(rul_sequence)
    decreasing_steps = np.sum(diffs <= 0)
    return decreasing_steps / len(diffs) if len(diffs) > 0 else 0.0

def calculate_oscillation(rul_sequence: np.ndarray) -> float:
    """RUL 예측의 진동(떨림) 정도를 계산합니다."""
    diffs = np.diff(rul_sequence)
    signs = np.sign(diffs)
    sign_changes = np.sum(signs[:-1] != signs[1:])
    return sign_changes / len(signs) if len(signs) > 0 else 0.0

def plot_rul_trajectory(time_axis, pred_rul, gt_rul, run_id, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # Smooth prediction (moving average)
    window = min(10, max(1, len(pred_rul) // 20))
    if len(pred_rul) >= window:
        smooth_pred = np.convolve(pred_rul, np.ones(window)/window, mode='valid')
        smooth_time = time_axis[window-1:]
        plt.plot(smooth_time, smooth_pred, label='Pred RUL (Smoothed)', color='orange', linewidth=2)
    
    plt.plot(time_axis, pred_rul, label='Pred RUL (Raw)', color='red', alpha=0.4, linewidth=1)
    plt.plot(time_axis, gt_rul, label='Ground Truth RUL', color='blue', linestyle='--', linewidth=2)
    
    mono = calculate_monotonicity(pred_rul)
    osc = calculate_oscillation(pred_rul)
    
    plt.title(f"RUL Trajectory - Run {run_id} | Monotonicity: {mono:.3f} | Oscillation: {osc:.3f}")
    plt.xlabel("Time")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.close('all')

def plot_prediction_error_heatmap(time_axis, pred_rul, gt_rul, save_path=None):
    """
    시간에 따른 Prediction Error (abs(pred - gt))를 Heatmap 형태로 시각화합니다.
    """
    error = np.abs(pred_rul - gt_rul)
    
    # 2D Heatmap 형태로 변환하기 위해 차원 확장 (1, T)
    error_2d = error[np.newaxis, :]
    
    plt.figure(figsize=(12, 4))
    
    # imshow를 활용해 1D Heatmap(차원 불일치 문제 해결) 적용
    plt.imshow(error_2d, cmap='hot', aspect='auto', extent=[time_axis[0], time_axis[-1], 0, 1], origin='lower')
    plt.colorbar(label='Absolute Error')
    
    # 시간에 따른 오차 추이를 선으로 오버레이
    plt.plot(time_axis, error / error.max() * 0.8 + 0.1, color='blue', alpha=0.5, linewidth=1.5, label='Normalized Error Trend')
    
    plt.title("Prediction Error Over Time")
    plt.xlabel("Time")
    plt.yticks([])
    plt.legend(loc='upper right')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.close('all')
