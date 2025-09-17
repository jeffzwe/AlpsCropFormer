swiss_class_names = {
        0: 'Background',  # Assuming class 0 is background
        1: 'SummerBarley',
        2: 'WinterBarley',
        3: 'Oat',
        4: 'Wheat',
        5: 'Grain',
        6: 'Maize',
        7: 'EinkornWheat',
        8: 'SummerWheat',
        9: 'WinterWheat',
        10: 'Rye',
        11: 'Spelt',
        12: 'Sugar_beets',
        13: 'Beets',
        14: 'Potatoes',
        15: 'SummerRapeseed',
        16: 'WinterRapeseed',
        17: 'Soy',
        18: 'Sunflowers',
        19: 'Linen',
        20: 'Hemp',
        21: 'Field bean',
        22: 'Peas',
        23: 'Lupine',
        24: 'Pumpkin',
        25: 'Tobacco',
        26: 'Sorghum',
        27: 'Vegetables',
        28: 'Chicory',
        29: 'Buckwheat',
        30: 'Berries',
        31: 'Void label',
        32: 'Biodiversity encouragement area',
        33: 'Fallow',
        34: 'MixedCrop',
        35: 'Mustard',
        36: 'Meadow',
        37: 'Pasture',
        38: 'Legumes',
        39: 'Vines',
        40: 'Apples',
        41: 'Pears',
        42: 'StoneFruit',
        43: 'Hops',
        44: 'TreeCrop',
        45: 'Chestnut',
        46: 'Special cultures',
        47: 'Hedge',
        48: 'Multiple',
        49: 'Forest',
        50: 'Non agriculture',
        51: 'Waters',
        52: 'Gardens'
    }

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_confusion_text(path: Path) -> Dict[int, Dict[int, float]]:
    """
    Parse a confusion.txt file that contains sections like:
      True class X:
        Top k: predicted as class Y (Z times, P%) (correct|misclassified)
        Top k: predicted as class Y P%
    Returns a mapping: true_class_id -> {pred_class_id: percent_value}
    Percent values are in [0, 100].
    """
    true_to_preds: Dict[int, Dict[int, float]] = {}
    current_true: int = None

    # Regex patterns
    header_pat = re.compile(r"^\s*True class\s+(\d+)(?::|\s*\(total samples:\s*0\):)")
    # Updated pattern to handle both formats: "85.5%" and "(Z times, P%)"
    pred_pat = re.compile(r"^\s*Top\s+\d+:\s+predicted as class\s+(\d+).*?(\d+(?:\.\d+)?)%")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            m_header = header_pat.match(line)
            if m_header:
                current_true = int(m_header.group(1))
                true_to_preds.setdefault(current_true, {})
                continue

            if current_true is None:
                continue

            m_pred = pred_pat.match(line)
            if m_pred:
                pred_cls = int(m_pred.group(1))
                pct = float(m_pred.group(2))
                # Accumulate in case of duplicates
                true_to_preds[current_true][pred_cls] = true_to_preds[current_true].get(pred_cls, 0.0) + pct

    return true_to_preds


def build_subset_matrix(
    parsed: Dict[int, Dict[int, float]],
    class_ids: List[int],
    renormalize: bool = False
) -> np.ndarray:
    """
    Build a (len(class_ids) x len(class_ids)) matrix of percentages.
    - Rows are true classes (in the provided order).
    - Columns are predicted classes (in the provided order).
    - Values are percentages from parsed data; predictions not in class_ids are ignored.
    - If renormalize=True, re-scale each row so the kept columns sum to 100% if the row has any mass.
    """
    n = len(class_ids)
    idx = {cid: i for i, cid in enumerate(class_ids)}
    mat = np.zeros((n, n), dtype=float)

    for t_cls, preds in parsed.items():
        if t_cls not in idx:
            continue
        r = idx[t_cls]
        for p_cls, pct in preds.items():
            if p_cls in idx:
                c = idx[p_cls]
                mat[r, c] += pct

    if renormalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        # Avoid divide by zero
        nonzero = row_sums.squeeze() > 0
        mat[nonzero] = mat[nonzero] / row_sums[nonzero] * 100.0

    return mat


def class_labels(class_ids: List[int]) -> List[str]:
    """
    Map class ids to names using swiss_class_names; fallback to str(id).
    """
    labels = []
    for cid in class_ids:
        name = swiss_class_names.get(cid, str(cid))
        labels.append(name)
    return labels


def plot_confusion_matrix(
    mat: np.ndarray,
    class_ids: List[int],
    outfile: Path,
    title: str = "Confusion matrix (Top-5 only)",
    annotate: bool = False,
    cmap: str = "viridis"
) -> None:
    """
    Plot a heatmap of the matrix (values in percentages) and save/show.
    """
    labels = class_labels(class_ids)
    
    # Define majority classes
    majority_classes = {0, 2, 6, 9, 16, 27, 36, 37}
    
    # Find separation points
    n_majority = sum(1 for cid in class_ids if cid in majority_classes)
    
    plt.figure(figsize=(max(10, len(class_ids) * 0.8), max(8, len(class_ids) * 0.7)))
    
    from matplotlib.colors import PowerNorm
    
    # Use power normalization - gamma between 0.5-1.0 for balance
    # gamma=0.7 gives a nice balance between linear and log scale
    ax = sns.heatmap(
        mat,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        cbar=True,
        norm=PowerNorm(gamma=0.65),  # Adjust this value: 0.5 (more like log) to 1.0 (linear)
        vmin=0.0,
        vmax=100.0,
        annot=annotate,
        fmt=".1f" if annotate else "",
        linewidths=0.5,
        linecolor='white'
    )
    # ax = sns.heatmap(
    #     mat,
    #     xticklabels=labels,
    #     yticklabels=labels,
    #     cmap=cmap,
    #     cbar=True,
    #     vmin=0.0,
    #     vmax=100.0,
    #     annot=annotate,
    #     fmt=".1f" if annotate else "",
    #     linewidths=0.5,
    #     linecolor='white'
    # )
    
    # Add thick lines to separate majority and minority classes
    if n_majority > 0 and n_majority < len(class_ids):
        # Vertical line
        ax.axvline(x=n_majority, color='red', linewidth=3)
        # Horizontal line
        ax.axhline(y=n_majority, color='red', linewidth=3)
    
    # Add text labels for groups (only top labels)
    if n_majority > 0 and n_majority < len(class_ids):
        # Majority classes label (top)
        ax.text(n_majority/2, -0.5, 'Majority Classes', ha='center', va='top', 
                fontsize=12, fontweight='bold', color='darkblue')
        # Minority classes label (top)
        ax.text(n_majority + (len(class_ids) - n_majority)/2, -0.5, 'Minority Classes', 
                ha='center', va='top', fontsize=12, fontweight='bold', color='darkred')
    
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, pad=45, fontsize=15)  # Add padding to move title further up
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        # Headless environments
        pass
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build and plot a subset confusion matrix (percentages) from confusion.txt")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/plots/confusion_strongweights.txt"),
        help="Path to confusion.txt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/plots/confusion_subset.png"),
        help="Output PNG file for the plotted matrix"
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Renormalize each row over the kept columns to sum to 100%%"
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="Annotate cells with percentage values"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[0, 2, 6, 9, 16, 27, 36, 37, 1, 3, 5, 12, 18, 24, 29, 40],  # Majority classes first, then minority
        help="Subset of class IDs to include (order matters)"
    )
    args = parser.parse_args()

    parsed = parse_confusion_text(args.input)
    mat = build_subset_matrix(parsed, args.classes, renormalize=args.renormalize)

    plot_confusion_matrix(
        mat,
        args.classes,
        outfile=args.output,
        title="Confusion Submatrix: Majority vs Minority Classes. Strong class weights.",
        annotate=args.annot
    )


if __name__ == "__main__":
    main()