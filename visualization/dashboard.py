from pathlib import Path
import numpy as np
import warnings
from .tsne import visualize_tsne, visualize_umap
from .embedding import extract_model_features
from .trajectory import plot_rul_trajectory, plot_prediction_error_heatmap
from .spectrum import plot_degradation_spectrum_evolution, plot_spectrogram

def save_epoch_dashboard(epoch, model, val_loader, device, run_ids=None, output_dir="outputs/visualization"):
    """Train Loop의 Epoch 끝단에 연결하여 자동 저장하는 통합 시스템."""
    save_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Visualization] Saving physics-aware dashboard for Epoch {epoch}...")
    
    try:
        # 1. Feature Extraction & t-SNE / UMAP
        features, gt_rul, pred_rul, attn_weights = extract_model_features(model, val_loader, device, feature_type="projection")
        
        if run_ids is None or len(run_ids) != len(gt_rul):
            run_ids = np.zeros(len(gt_rul))
            
        visualize_tsne(features, gt_rul.flatten(), run_ids, method="tsne", save_path=save_dir / "tsne_projection.png")
        visualize_umap(features, gt_rul.flatten(), run_ids, save_path=save_dir / "umap_projection.png")
        
        # 2. Trajectory & Error Heatmap (run_id 별로)
        unique_runs = np.unique(run_ids)
        for run_id in unique_runs:
            idx = (run_ids == run_id)
            if np.sum(idx) > 10:
                time_axis = np.arange(np.sum(idx))
                gt_sub = gt_rul.flatten()[idx]
                pred_sub = pred_rul.flatten()[idx]
                
                # RUL을 역순(시간순)으로 정렬
                sort_idx = np.argsort(gt_sub)[::-1]
                gt_sub, pred_sub = gt_sub[sort_idx], pred_sub[sort_idx]
                
                plot_rul_trajectory(time_axis, pred_sub, gt_sub, run_id=run_id, save_path=save_dir / f"trajectory_{run_id}.png")
                plot_prediction_error_heatmap(time_axis, pred_sub, gt_sub, save_path=save_dir / f"error_heatmap_{run_id}.png")
        
        # 3. Spectrum Evolution & Spectrogram (Mock or extracted if raw available)
        # TODO: DataLoader에서 raw vibration data를 받아와야 정확하게 plot 가능
        # 현재는 Placeholder로 남겨두며, 필요시 X_vib에서 STFT magnitude 복원 후 사용
        
        # 4. Attention Visualization
        if attn_weights is not None:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # (batch, seq_len, 1) -> (batch, seq_len)
            attn_weights_sq = np.squeeze(attn_weights, axis=-1)
            mean_attn = np.mean(attn_weights_sq, axis=0)
            
            # Entropy calculation: -sum(p * log(p)) over seq_len
            entropy = -np.sum(mean_attn * np.log(mean_attn + 1e-9))
            
            plt.figure(figsize=(10, 4))
            sns.barplot(x=np.arange(len(mean_attn)), y=mean_attn, color='blue')
            plt.title(f"Average Temporal Attention Weights (Entropy: {entropy:.3f})")
            plt.xlabel("Timestep")
            plt.ylabel("Attention Weight")
            plt.tight_layout()
            plt.savefig(save_dir / "attention_weights.png", dpi=150)
            plt.close()
        
        print(f"[Visualization] Dashboard saved to {save_dir}")
        
    except Exception as e:
        print(f"[Visualization] Error during dashboard generation: {e}")
        import traceback
        traceback.print_exc()
