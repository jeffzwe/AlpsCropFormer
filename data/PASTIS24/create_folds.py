import os
import glob
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import KFold

def create_folds(data_dir, output_dir, n_folds=5, seed=42):
    """
    Create fold splits from pickle files and save as CSV files.
    
    Args:
        data_dir: Directory containing pickle files
        output_dir: Directory to save CSV files
        n_folds: Number of folds to create
        seed: Random seed for reproducibility
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pickle files in the directory
    pickle_files = glob.glob(os.path.join(data_dir, "*.pickle"))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {data_dir}")
    
    # Get the base folder name for the pickle files
    pickle_folder_name = os.path.basename(data_dir)
    
    # Create filenames with the folder prefix
    filenames = [os.path.join(pickle_folder_name, os.path.basename(f)) for f in pickle_files]
    
    # Shuffle the files with a fixed seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(filenames)
    
    # Create KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Dictionary to store fold indices
    fold_indices = {}
    
    # Generate folds and save individual fold files
    for fold_idx, (_, test_indices) in enumerate(kf.split(filenames), 1):
        fold_name = f"fold{fold_idx}"
        fold_files = [filenames[i] for i in test_indices]
        fold_indices[fold_idx] = test_indices
        
        # Save individual fold
        df = pd.DataFrame(fold_files, columns=["filename"])
        output_path = os.path.join(output_dir, f"{fold_name}.csv")
        df.to_csv(output_path, index=False, header=False)
        print(f"Created {output_path} with {len(fold_files)} files")
    
    # Generate aggregated folds
    aggregations = [
        ("folds_123", [1, 2, 3]),
        ("folds_234", [2, 3, 4]),
        ("folds_345", [3, 4, 5]),
        ("folds_451", [4, 5, 1]),
        ("folds_512", [5, 1, 2])
    ]
    
    for agg_name, fold_list in aggregations:
        agg_indices = []
        for fold in fold_list:
            agg_indices.extend(fold_indices[fold])
        
        agg_files = [filenames[i] for i in agg_indices]
        df = pd.DataFrame(agg_files, columns=["filename"])
        output_path = os.path.join(output_dir, f"{agg_name}.csv")
        df.to_csv(output_path, index=False, header=False)
        print(f"Created {output_path} with {len(agg_files)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create data fold splits for model training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing pickle files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV files")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds to create")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_folds(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        seed=args.seed
    )
