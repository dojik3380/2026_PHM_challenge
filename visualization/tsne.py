import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def visualize_tsne(features, rul_labels, run_ids, method="tsne", save_path=None):
    # Feature 자동 Flatten
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
        
    # 대용량 Feature 자동 Subsampling (OOM 방지)
    max_samples = 3000
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        rul_labels = rul_labels[indices]
        run_ids = run_ids[indices]
        
    if method.lower() == "tsne":
        # Perplexity는 반드시 샘플 수보다 작아야 합니다. (기본값 30)
        perplexity = min(30, max(2, len(features) - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    elif method.lower() == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
        
    embeddings = reducer.fit_transform(features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. RUL Color Map (Degradation Continuity)
    scatter1 = ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=rul_labels, cmap='viridis', alpha=0.7, s=20, zorder=2)
    ax1.set_title(f"Latent Manifold by RUL ({method.upper()})")
    fig.colorbar(scatter1, ax=ax1, label='RUL')
    
    # 2. Run ID Color Map (Run Memorization 확인)
    unique_runs = np.unique(run_ids)
    run_id_map = {run: idx for idx, run in enumerate(unique_runs)}
    run_ids_numeric = np.array([run_id_map[r] for r in run_ids])
    
    cmap = plt.cm.get_cmap('tab10', len(unique_runs))
    scatter2 = ax2.scatter(embeddings[:, 0], embeddings[:, 1], c=run_ids_numeric, cmap=cmap, alpha=0.7, s=20, zorder=2)
    ax2.set_title(f"Latent Manifold by Run ID ({method.upper()})")
    cbar2 = fig.colorbar(scatter2, ax=ax2, ticks=np.arange(len(unique_runs)))
    cbar2.set_ticklabels(unique_runs)
    cbar2.set_label('Run ID')
    
    # Latent Trajectory (점들을 선으로 연결)
    for run in unique_runs:
        idx = (run_ids == run)
        # RUL이 높은(초기) -> 낮은(후기) 순서로 정렬하여 선으로 연결
        # 데이터가 이미 시간 순서라고 가정하지만, 확실히 하기 위해 RUL 역순 정렬
        run_rul = rul_labels[idx]
        run_emb = embeddings[idx]
        sort_idx = np.argsort(run_rul)[::-1] 
        ax2.plot(run_emb[sort_idx, 0], run_emb[sort_idx, 1], color=cmap(run_id_map[run]), alpha=0.3, linewidth=1, zorder=1)
        ax1.plot(run_emb[sort_idx, 0], run_emb[sort_idx, 1], color='gray', alpha=0.2, linewidth=1, zorder=1)
    
    # Run Memorization 위험도 판별 로직
    run_centers = [embeddings[run_ids == run_id].mean(axis=0) for run_id in unique_runs]
    if len(unique_runs) > 1:
        overall_spread = np.std(embeddings, axis=0).mean()
        center_spread = np.std(run_centers, axis=0).mean()
        # 중심간 거리가 전체 퍼짐도보다 크면 분리된 섬(Island)으로 판단
        if center_spread > overall_spread * 0.8:
            warnings.warn("High risk of Run Memorization: Island clusters detected by Run ID.")
            fig.text(0.5, 0.01, "WARNING: Possible Run Memorization Detected (Island Clusters)", ha='center', color='red', fontsize=12, fontweight='bold')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.close('all')

def visualize_umap(features, rul_labels, run_ids, save_path=None):
    if not HAS_UMAP:
        print("UMAP is not installed. Please install umap-learn to use this feature.")
        return
    visualize_tsne(features, rul_labels, run_ids, method="umap", save_path=save_path)

