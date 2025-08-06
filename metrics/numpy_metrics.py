import numpy as np
from sklearn.metrics import confusion_matrix


def get_prediction_splits(predicted, labels, n_classes):
    cm = confusion_matrix(labels, predicted, labels=np.arange(n_classes)).astype(np.float32)
    diag = np.diagonal(cm)
    rowsum = cm.sum(axis=1)
    colsum = cm.sum(axis=0)
    TP = (diag).astype(np.float32)
    FN = (rowsum - diag).astype(np.float32)
    FP = (colsum - diag).astype(np.float32)
    IOU = diag / (rowsum + colsum - diag)
    micro_IOU = diag.sum() / (rowsum.sum() + colsum.sum() - diag.sum())

    num_total = []
    num_correct = []
    for class_ in range(n_classes):
        idx = labels == class_
        is_correct = predicted[idx] == labels[idx]
        if is_correct.size == 0:
            is_correct = np.array(0)
        num_total.append(idx.sum())
        num_correct.append(is_correct.sum())   # previously was .mean()
    num_total = np.array(num_total).astype(np.float32)
    num_correct = np.array(num_correct)

    return TP, FP, FN, num_correct, num_total, IOU, micro_IOU


def get_metrics_from_splits(TP, FP, FN, num_correct, num_total):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if (type(precision) in [np.float32, np.float64]) and (precision + recall == 0.0):
        F1 = 0.0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    acc = num_correct / num_total
    return acc, precision, recall, F1


def nan_mean(v):
    return v[~np.isnan(v)].mean()


def get_classification_metrics(predicted, labels, n_classes, unk_masks=None):
    if unk_masks is not None:
        predicted = predicted[unk_masks]
        labels = labels[unk_masks]
    TP, FP, FN, num_correct, num_total, IOU, micro_IOU = get_prediction_splits(predicted, labels, n_classes) #  , per_class)
    micro_acc, micro_precision, micro_recall, micro_F1 = \
        get_metrics_from_splits(TP.sum(), FP.sum(), FN.sum(), num_correct.sum(), num_total.sum())
    macro_IOU = IOU[~np.isnan(IOU)].mean()
    acc, precision, recall, F1 = \
        get_metrics_from_splits(TP, FP, FN, num_correct, num_total)
    macro_acc = nan_mean(acc)
    macro_precision = nan_mean(precision)
    macro_recall = nan_mean(recall)
    macro_F1 = nan_mean(F1)
    acc = np.nan_to_num(acc, copy=True, nan=0.0)
    precision = np.nan_to_num(precision, copy=True, nan=0.0)
    recall = np.nan_to_num(recall, copy=True, nan=0.0)
    F1 = np.nan_to_num(F1, copy=True, nan=0.0)
    IOU = np.nan_to_num(IOU, copy=True, nan=0.0)
    return {'class': [acc, precision, recall, F1, IOU],
            'micro': [micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU],
            'macro': [macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU]}



def get_per_class_loss(losses, labels, unk_masks=None):
    # print(f"Shapes: losses: {losses.shape}, labels: {labels.shape}, unk_masks: {unk_masks.shape if unk_masks is not None else 'None'}")
    if unk_masks is not None:
        losses = losses[unk_masks]
        labels = labels[unk_masks]
    unique_labels = np.unique(labels)
    class_loss = []
    for label in unique_labels:
        idx = labels == label
        class_loss.append(losses[idx].mean())
    return unique_labels, np.asarray(class_loss)


def get_top_predictions_per_class(predicted, labels, n_classes, top_x=5):
    """
    For each true class, find the top x most frequently predicted classes
    (including the correct class) and their frequencies.
    
    Args:
        predicted: array of predicted labels (already filtered)
        labels: array of true labels (already filtered)
        n_classes: total number of classes
        top_x: number of top predictions to return
    
    Returns:
        dict: For each class, contains the top predicted classes and their counts/percentages
    """
    # cm = confusion_mat(predicted, labels, n_classes).astype(np.float32)
    cm = confusion_matrix(labels, predicted, labels=np.arange(n_classes)).astype(np.float32)
    
    top_predictions = {}
    
    for true_class in range(n_classes):
        # Get all predictions for this true class (row in confusion matrix)
        predictions_for_class = cm[true_class, :]
        
        # Get top x predicted classes
        top_indices = np.argsort(predictions_for_class)[::-1][:top_x]
        
        # Filter out classes with zero predictions
        top_indices = top_indices[predictions_for_class[top_indices] > 0]
        
        top_counts = predictions_for_class[top_indices]
        total_samples = cm[true_class, :].sum()
        
        # Calculate percentages
        top_percentages = (top_counts / total_samples * 100) if total_samples > 0 else np.zeros_like(top_counts)
        
        top_predictions[true_class] = {
            'top_classes': top_indices.tolist(),
            'counts': top_counts.tolist(),
            'percentages': top_percentages.tolist(),
            'total_samples': int(total_samples)
        }
    
    return top_predictions