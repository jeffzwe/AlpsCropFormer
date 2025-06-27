import pickle
from pathlib import Path
import tqdm
import torch

def preprocess_and_save_dataset(dataset, save_dir):
    """
    Iterate through dataset and save each sample as individual pickle files
    
    Args:
        dataset: Your Sentinel2Dataset instance
        save_dir: Directory to save preprocessed pickle files
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing and saving {len(dataset)} samples to {save_path}")
    
    sample_counter = 0
    for idx in tqdm.tqdm(range(len(dataset)), desc="Preprocessing and saving"):
        inputs, unk_masks, labels = dataset[idx]
        
        # Iterate through each element in the lists
        for i in range(len(inputs)):
            sample_data = {
                'inputs': torch.from_numpy(inputs[i].copy()).float(),
                'unk_masks': torch.from_numpy(unk_masks[i].copy()).unsqueeze(-1),
                'labels': torch.from_numpy(labels[i].copy()).float().unsqueeze(-1)
            }
            
            with open(save_path / f'sample_{sample_counter:06d}.pkl', 'wb') as f:
                pickle.dump(sample_data, f)
            
            sample_counter += 1
    
    print(f"Saved {sample_counter} preprocessed samples to {save_path}")