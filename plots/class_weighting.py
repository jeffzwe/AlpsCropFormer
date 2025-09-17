import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import seaborn as sns

def read_tensorboard_scalar(log_dir, scalar_tag):
    """Read scalar values from TensorBoard event files."""
    values = []
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    for event_file in event_files:
        try:
            for event in summary_iterator(event_file):
                for value in event.summary.value:
                    if value.tag == scalar_tag:
                        values.append(value.simple_value)
        except Exception as e:
            print(f"Error reading {event_file}: {e}")
    
    return values

def main():
    # Define class reweighting factors
    alpha_values = [0.0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0]
    
    # Base path for model logs
    base_path = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/models/model_logs/Sentinel"
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return
    
    # Storage for results
    recall_macro_max_values = []
    accuracy_micro_max_values = []
    precision_macro_max_values = []
    valid_alphas = []
    
    for alpha in alpha_values:
        # Construct experiment path with uniform two decimal formatting
        exp_name = f"TSViT_fold1.sliding_window.focal.gamma2.0.alpha_adaptive{alpha:.2f}"
        exp_path = os.path.join(base_path, exp_name)
        
        if not os.path.exists(exp_path):
            continue
        
        # Read Recall, Precision and Accuracy values
        recall_macro_log_dir = os.path.join(exp_path, "Recall_eval_macro_Average")
        accuracy_micro_log_dir = os.path.join(exp_path, "Accuracy_eval_micro_Average")
        precision_macro_log_dir = os.path.join(exp_path, "Precision_eval_macro_Average")
        
        recall_macro_values = read_tensorboard_scalar(recall_macro_log_dir, "Recall")
        accuracy_micro_values = read_tensorboard_scalar(accuracy_micro_log_dir, "Accuracy")
        precision_macro_values = read_tensorboard_scalar(precision_macro_log_dir, "Precision")
        
        if recall_macro_values and accuracy_micro_values and precision_macro_values:
            recall_macro_max_values.append(max(recall_macro_values))
            accuracy_micro_max_values.append(max(accuracy_micro_values))
            precision_macro_max_values.append(max(precision_macro_values))
            valid_alphas.append(alpha)
            print(f"Alpha {alpha}: Recall={max(recall_macro_values):.4f}, Precision={max(precision_macro_values):.4f}, Accuracy={max(accuracy_micro_values):.4f}")
        else:
            print(f"No valid data found for alpha {alpha}")
    
    # Check if we have any valid data
    if not valid_alphas:
        print("No valid data found for any alpha values. Cannot create plot.")
        return
    
    # Create the plot
    sns.set_style("whitegrid")
    sns.set_palette("viridis")
    plt.figure(figsize=(12, 8))
    
    # Plot macro recall, macro precision, and micro accuracy using seaborn
    sns.lineplot(x=valid_alphas, y=recall_macro_max_values, marker='o', label='Macro Average Recall', linewidth=2, markersize=8)
    sns.lineplot(x=valid_alphas, y=precision_macro_max_values, marker='^', label='Macro Average Precision', linewidth=2, markersize=8)
    sns.lineplot(x=valid_alphas, y=accuracy_micro_max_values, marker='s', label='Micro Average Accuracy', linewidth=2, markersize=8)
    
    plt.xlabel('Class Reweighting Factor (α)', fontsize=12)
    plt.ylabel('Maximum Score', fontsize=12)
    plt.title('Maximum Recall, Precision and Accuracy vs Class Reweighting Factor\n(TSViT with Sliding Window, Focal Loss γ=2.0)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show all alpha values
    plt.xticks(valid_alphas)
    
    # Add value annotations on points
    for i, (alpha, recall_macro_val, precision_macro_val, accuracy_micro_val) in enumerate(
        zip(valid_alphas, recall_macro_max_values, precision_macro_max_values, accuracy_micro_max_values)):
        plt.annotate(f'{recall_macro_val:.3f}', (alpha, recall_macro_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{precision_macro_val:.3f}', (alpha, precision_macro_val), textcoords="offset points", 
                    xytext=(5,5), ha='center', fontsize=8)
        plt.annotate(f'{accuracy_micro_val:.3f}', (alpha, accuracy_micro_val), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/Users/jeffreyzweidler/Desktop/Semester_Project/AlpsCropFormer/plots/recall_precision_accuracy_class_weighting_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    if recall_macro_max_values and precision_macro_max_values and accuracy_micro_max_values:
        print("\nSummary:")
        print(f"Best Macro Recall: {max(recall_macro_max_values):.4f} at α={valid_alphas[np.argmax(recall_macro_max_values)]}")
        print(f"Best Macro Precision: {max(precision_macro_max_values):.4f} at α={valid_alphas[np.argmax(precision_macro_max_values)]}")
        print(f"Best Micro Accuracy: {max(accuracy_micro_max_values):.4f} at α={valid_alphas[np.argmax(accuracy_micro_max_values)]}")
    else:
        print("No data to summarize.")

if __name__ == "__main__":
    main()