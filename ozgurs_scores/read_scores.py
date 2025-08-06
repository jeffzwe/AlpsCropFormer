import pickle as pkl
import numpy as np
import pandas as pd

# Load the names from label sheet (integrated from confusionMatrix.py)
label_sheet = pd.read_csv("crop_mappings.csv")
csv_key = '4th_tier_ENG'
mapping_dict = {}
unique_codes = label_sheet[csv_key].unique()
for idx, code in enumerate(unique_codes):
    mapping_dict[code] = idx + 1
target_mapping = {0: 0, -1: 0}
for _, row in label_sheet.iterrows():
    target_mapping[int(row['LNF_code'])] = mapping_dict[row[csv_key]]
num_classes = len(unique_codes) + 1
sorted_items = sorted(mapping_dict.items(), key=lambda x: x[1])
labels = [k for k, _ in sorted_items]

def prepare_cm(path):
  cm = pkl.load(open(path, "rb"))
  cm = np.delete(cm, 0, axis=0)
  cm = np.delete(cm, 0, axis=1)
  return cm
 
def confusion_matrix_analysis(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        mat (array): confusion matrix
 
    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics
 
    """
    TP = 0
    FP = 0
    FN = 0
 
    per_class = {}
 
    for j in range(mat.shape[0]):
        d = {}
        tp = np.nansum(mat[j, j])
        fp = np.nansum(mat[:, j]) - tp
        fn = np.nansum(mat[j, :]) - tp
 
        d['IoU'] = tp / (tp + fp + fn + 1e-6)
        d['Precision'] = tp / (tp + fp + 1e-6)
        d['Recall'] = tp / (tp + fn + 1e-6)
        d['F1-score'] = 2 * tp / ((2 * tp + fp + fn) + 1e-6)
 
        per_class[str(j)] = d
 
        TP += tp
        FP += fp
        FN += fn
 
    overall = {}
    overall['micro_IoU'] = TP / (TP + FP + FN)
    overall['micro_Precision'] = TP / (TP + FP)
    overall['micro_Recall'] = TP / (TP + FN)
    overall['micro_F1-score'] = 2 * TP / (2 * TP + FP + FN)
 
    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = macro.loc['IoU']
    overall['MACRO_Precision'] = macro.loc['Precision']
    overall['MACRO_Recall'] = macro.loc['Recall']
    overall['MACRO_F1-score'] = macro.loc['F1-score']
 
    overall['Accuracy'] = np.sum(np.diag(mat)) / np.sum(mat)
 
    return per_class, overall

def main():
    # Read and prepare the confusion matrix
    cm_path = "ozgurs_scores/conf_mat.pkl"
    cm = prepare_cm(cm_path)
    
    # Perform confusion matrix analysis
    per_class, overall = confusion_matrix_analysis(cm)
    
    # Print results
    print("=== CONFUSION MATRIX ANALYSIS ===\n")
    
    print("Per-class metrics:")
    print("-" * 50)
    for class_id, metrics in per_class.items():
        class_idx = int(class_id)
        class_name = labels[class_idx] if class_idx < len(labels) else f"Class_{class_idx}"
        print(f"Class {class_idx + 1} ({class_name}):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    print("Overall metrics:")
    print("-" * 50)
    for metric, value in overall.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()


