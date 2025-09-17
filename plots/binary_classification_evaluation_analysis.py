import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.distributed as dist

# Set seaborn style for publication-ready plots
sns.set_style("whitegrid")
sns.set_palette("viridis")

def evaluate_patch_and_pixel_npv_with_confusion(logits_all, labels_all, threshold, patch_size=(16, 16)):
    """
    Evaluates patch-level confusion matrix and pixel-level NPV.

    Args:
        logits_all: 1D numpy array of logits (flattened per-pixel outputs)
        labels_all: 1D numpy array of ground truth labels (binary)
        threshold: float, sigmoid threshold for predicting class 1
        patch_size: tuple (H, W) size of each patch

    Returns:
        Dictionary with:
            - n_true_bg_patches
            - n_pred_bg_patches
            - pixelwise_npv
            - patch_TP, patch_TN, patch_FP, patch_FN
    """
    assert len(logits_all) == len(labels_all)
    patch_area = patch_size[0] * patch_size[1]
    assert len(logits_all) % patch_area == 0, "Data does not divide evenly into patches"

    # ---- Pixel-wise NPV ----
    probs = 1 / (1 + np.exp(-logits_all))
    preds = (probs < threshold).astype(np.uint8)  # class 0 if prob < threshold

    tn = np.logical_and(preds == 0, labels_all == 0).sum()
    fn = np.logical_and(preds == 0, labels_all == 1).sum()
    pixelwise_npv = tn / (tn + fn + 1e-8)

    # ---- Patch-level confusion matrix ----
    logits_patches = logits_all.reshape(-1, patch_area)
    labels_patches = labels_all.reshape(-1, patch_area)

    probs_patches = 1 / (1 + np.exp(-logits_patches))
    pred_patches = (probs_patches >= threshold).any(axis=1)  # True = predicted class 1 exists
    true_patches = (labels_patches >= 0.5).any(axis=1)       # True = class 1 exists

    # Confusion matrix components
    patch_TP = np.logical_and(pred_patches == True, true_patches == True).sum()
    patch_TN = np.logical_and(pred_patches == False, true_patches == False).sum()
    patch_FP = np.logical_and(pred_patches == True, true_patches == False).sum()
    patch_FN = np.logical_and(pred_patches == False, true_patches == True).sum()

    # For backwards compatibility
    n_true_bg_patches = (~true_patches).sum()
    n_pred_bg_patches = (~pred_patches).sum()

    return {
        'threshold': threshold,
        'n_true_bg_patches': int(n_true_bg_patches),
        'n_pred_bg_patches': int(n_pred_bg_patches),
        'pixelwise_npv': float(pixelwise_npv),
        'patch_TP': int(patch_TP),
        'patch_TN': int(patch_TN),
        'patch_FP': int(patch_FP),
        'patch_FN': int(patch_FN),
    }

def evaluate(net, evalloader, loss_fn, device, loss_input_fn, config, is_distributed=False):
    """Evaluation function adapted for binary classification with BCEWithLogitsLoss"""
    labels_all = []
    losses_all = []
    logits_all = []

    rank = dist.get_rank() if is_distributed else -1
    net.eval()
    sample_count = 0

    with torch.no_grad():
        for sample in evalloader:
            sample_count += 1
            inputs = sample['inputs'].to(device)
            logits = net(inputs)

            ground_truth = loss_input_fn(sample, device)
            
            logits = logits.squeeze(1)

            loss = loss_fn['all'](logits, ground_truth)
            losses_all.append(loss.detach().cpu().flatten().numpy())

            # Store logits and labels for metrics later
            logits_all.append(logits.detach().cpu().flatten().numpy())
            labels_all.append(ground_truth.detach().cpu().flatten().numpy())

    print(f"[Rank {rank}] Evaluation complete with sample count: {sample_count}")

    # Gather results from all processes if distributed
    if is_distributed:
        local_data = {
            'logits': np.concatenate(logits_all),
            'labels': np.concatenate(labels_all),
            'losses': np.concatenate(losses_all)
        }
        gathered_data = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_data, local_data)

        if dist.get_rank() == 0:
            logits_all = np.concatenate([d['logits'] for d in gathered_data])
            labels_all = np.concatenate([d['labels'] for d in gathered_data])
            losses = np.concatenate([d['losses'] for d in gathered_data])
        else:
            return
    else:
        logits_all = np.concatenate(logits_all)
        labels_all = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)
        
    thresholds = [0.999, 0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]
    for t in thresholds:
        result = evaluate_patch_and_pixel_npv_with_confusion(logits_all, labels_all, t)
        print(
            f"Threshold {t:.2f} → "
            f"Pixel-wise NPV: {result['pixelwise_npv']:.4f} | "
            f"TP: {result['patch_TP']}, "
            f"TN: {result['patch_TN']}, "
            f"FP: {result['patch_FP']}, "
            f"FN: {result['patch_FN']} | "
            f"True BG patches: {result['n_true_bg_patches']}, "
            f"Predicted BG patches: {result['n_pred_bg_patches']}"
        )
            
    return

def main():
    """Create a plot showing threshold vs True BG patches, Predicted BG patches, and FN"""
    
    # Data from the evaluation results
    data = {
        'Threshold': [1.00, 0.99, 0.95, 0.90, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.01, 0.00],
        'True_BG_patches': [424024] * 12,  # Constant value
        'Predicted_BG_patches': [1242241, 898265, 766749, 710612, 612832, 467942, 405819, 365474, 298631, 225877, 74636, 11215],
        'FN': [819151, 485031, 365421, 317344, 241693, 139197, 103768, 84259, 58132, 37540, 8509, 216]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    
    # Create the plot using seaborn
    sns.lineplot(data=df, x='Threshold', y='True_BG_patches', linestyle='--', linewidth=2, label='True Background Patches', alpha=0.8, color='black')
    sns.lineplot(data=df, x='Threshold', y='Predicted_BG_patches', linewidth=2, marker='o', label='Predicted Background Patches')
    sns.lineplot(data=df, x='Threshold', y='FN', linewidth=2, marker='s', label='False Negative Predictions (FN)')
    
    # Customize the plot
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Count (in thousands)', fontsize=12)
    plt.title('Binary classification background vs foreground', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Reverse x-axis to show decreasing thresholds
    plt.gca().invert_xaxis()
    
    # Format y-axis to show steps of 1000 in thousands
    ax = plt.gca()
    max_val = max(max(df['Predicted_BG_patches']), max(df['FN']), max(df['True_BG_patches']))
    y_ticks = np.arange(0, max_val + 200000, 200000)  # Steps of 200k
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(tick/1000)}' for tick in y_ticks])
    
    plt.tight_layout()
    
    output_path = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/plots/binary_background_foreground.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()


