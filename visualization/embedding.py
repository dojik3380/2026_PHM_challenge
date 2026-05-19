import torch
import numpy as np

def extract_model_features(model, dataloader, device, feature_type="lstm"):
    """
    feature_type: "cnn", "projection", "lstm"
    """
    model.eval()
    features = []
    labels = []
    preds = []
    attn_weights_list = []
    
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activation[name] = output[0].detach().cpu()
            else:
                activation[name] = output.detach().cpu()
        return hook
        
    hook_handle = None
    if feature_type == "cnn":
        hook_handle = model.vibration_cnn.register_forward_hook(get_activation(feature_type))
    elif feature_type == "projection":
        hook_handle = model.vibration_projection.register_forward_hook(get_activation(feature_type))
    elif feature_type == "lstm":
        hook_handle = model.vibration_lstm.register_forward_hook(get_activation(feature_type))
    
    with torch.no_grad():
        for batch in dataloader:
            batch_vib = batch[0].to(device)
            batch_op = batch[1].to(device)
            batch_y = batch[2].to(device)
            
            output = model(batch_vib, batch_op)
            
            if hasattr(model, 'last_attn_weights') and model.last_attn_weights is not None:
                attn_weights_list.append(model.last_attn_weights.numpy())
            
            if feature_type in activation:
                feat = activation[feature_type].numpy()
                batch_size = batch_y.shape[0]
                
                # CNN/Projection 같이 (B * seq_len, D) 형태로 나오는 경우 (B, seq_len, D)로 복원
                if feat.ndim == 2 and feat.shape[0] > batch_size and feat.shape[0] % batch_size == 0:
                    seq_len = feat.shape[0] // batch_size
                    feat = feat.reshape(batch_size, seq_len, -1)
                
                # 시퀀스 데이터인 경우 마지막 타임스텝 사용
                if feat.ndim == 3:
                    feat = feat[:, -1, :] 
                    
                features.append(feat)
                labels.append(batch_y.cpu().numpy().flatten())
                preds.append(output.cpu().numpy().flatten())
                
    if hook_handle:
        hook_handle.remove()
        
    attn_weights = np.concatenate(attn_weights_list, axis=0) if attn_weights_list else None
    return np.concatenate(features), np.concatenate(labels), np.concatenate(preds), attn_weights
