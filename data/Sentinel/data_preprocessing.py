import os
import sys
import numpy as np
import webdataset as wds
from tqdm import tqdm
import pandas as pd
import zarr
from torch.utils.data import Dataset

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from data.Sentinel.data_transforms import RemoveDuplicateTimestamps


class PreprocessingDataset(Dataset):
    def __init__(
        self,
        sentinel2_dir,
        gt_dir,
        temp_calendar_dir,
        label_sheet_file="./label_sheet.csv",
        bands=[
            's2_B02', 's2_B03', 's2_B04', 's2_B08',
            's2_B05', 's2_B06', 's2_B07', 's2_B8A', 's2_B12'
        ],
        patch_size=16,
        cloud_threshold=0.9,
        seed=42
    ):
        self.sentinel2_dir = sentinel2_dir
        self.gt_dir = gt_dir
        self.temp_calendar_dir = temp_calendar_dir
        self.label_sheet_file = label_sheet_file
        self.bands = bands
        self.patch_size = patch_size
        self.cloud_threshold = cloud_threshold
        
        if seed is not None:
            np.random.seed(seed)

        self.data_files = self._get_data_files()
        
        # Setup transforms
        self.remove_duplicates = RemoveDuplicateTimestamps()
        # Setup label mapping
        self.map_lnf_code_to_ground_truth()

    def _get_data_files(self):
        data_files = []
        gt_files = [f for f in os.listdir(self.gt_dir) if f.endswith('.zarr')]
        for gt_file in gt_files:
            data_file_path = os.path.join(self.sentinel2_dir, gt_file)
            temp_calendar_path = os.path.join(self.temp_calendar_dir, gt_file)
            if os.path.exists(data_file_path) and os.path.exists(temp_calendar_path):
                data_files.append((data_file_path, os.path.join(self.gt_dir, gt_file), temp_calendar_path))
        return data_files

    def __len__(self):
        return len(self.data_files)

    def map_lnf_code_to_ground_truth(self):
        label_sheet = pd.read_csv(self.label_sheet_file)
        csv_key = '4th_tier_ENG'
        mapping_dict = {}
        unique_codes = label_sheet[csv_key].unique()
        for idx, code in enumerate(unique_codes):
            mapping_dict[code] = idx + 1
        self.target_mapping = {0: 0, -1: 0}
        for _, row in label_sheet.iterrows():
            self.target_mapping[int(row['LNF_code'])] = mapping_dict[row[csv_key]]

    def __getitem__(self, idx):
        data_file, gt_file, temp_calendar_file = self.data_files[idx]
        
        # Load data
        with zarr.open(data_file, mode='r') as s2_data, \
             zarr.open(gt_file, mode='r') as gt_data:
            
            # Load S2 data
            bands_data = [s2_data[band][:].astype(np.float32) for band in self.bands]
            images = np.stack(bands_data, axis=0)  # C x T x H x W
            time_stamps = s2_data['/time'][:]
            cloud_mask = s2_data['s2_mask'][:].astype(np.int16)
            
            # Load GT
            lnf = gt_data['lnf_code'][:]
            lnf = np.where(lnf == None, 0, lnf).astype(np.int32)
            ground_truth = np.vectorize(lambda x: self.target_mapping.get(x, 0))(lnf).astype(np.int32)
        
        # Load temperature calendar
        with zarr.open(temp_calendar_file, mode='r') as temp_data:
            temp_cal = temp_data['temperature_calendar'][:].astype(np.float32)
        
        # Apply remove duplicates transform only
        sample = (images, time_stamps, cloud_mask, temp_cal, ground_truth)
        images, time_stamps, cloud_mask, temp_cal, ground_truth = self.remove_duplicates(sample)
        
        # Create patches from 128x128 to 16x16 - work with raw C x T x H x W format
        C, T, H, W = images.shape  # C x T x H x W
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Initialize lists to store valid patches
        valid_patches = []
        valid_cloud_masks = []
        valid_ground_truths = []
        valid_time_stamps = []
        
        # Extract patches and filter by cloud coverage
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                h_start = h * self.patch_size
                h_end = h_start + self.patch_size
                w_start = w * self.patch_size
                w_end = w_start + self.patch_size
                
                # Extract patch
                patch_images = images[:, :, h_start:h_end, w_start:w_end]  # C x T x patch_size x patch_size
                patch_cloud_mask = cloud_mask[:, h_start:h_end, w_start:w_end]  # T x patch_size x patch_size
                patch_gt = ground_truth[h_start:h_end, w_start:w_end]  # patch_size x patch_size
                
                # Filter timesteps with >90% cloud coverage
                valid_timesteps = []
                for t in range(T):
                    cloud_coverage = np.mean(patch_cloud_mask[t] == 1)
                    if cloud_coverage <= self.cloud_threshold:
                        valid_timesteps.append(t)
                
                if len(valid_timesteps) > 0:
                    # Keep only valid timesteps
                    patch_images = patch_images[:, valid_timesteps, :, :]  # C x valid_T x patch_size x patch_size
                    patch_cloud_mask = patch_cloud_mask[valid_timesteps, :, :]
                    patch_time_stamps = time_stamps[valid_timesteps]
                    
                    valid_patches.append(patch_images)
                    valid_cloud_masks.append(patch_cloud_mask)
                    valid_ground_truths.append(patch_gt)
                    valid_time_stamps.append(patch_time_stamps)
        
        if len(valid_patches) == 0:
            # Return empty arrays if no valid patches
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        
        return (valid_patches, valid_time_stamps, valid_cloud_masks, valid_ground_truths, temp_cal)


def save_to_webdataset(dataset, output_dir, samples_per_shard=10000, max_samples=None):
    """
    Save samples from a Dataset to WebDataset tar files.

    Args:
        dataset (Dataset): PyTorch Dataset yielding samples.
        output_dir (str): Directory on scratch where WebDataset shards will be saved.
        samples_per_shard (int): Number of samples per tar shard.
        max_samples (int, optional): Max number of total samples to save. Useful for debugging or partial export.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    sample_id = 0
    shard_id = 0
    sink = None

    try:
        for idx in tqdm(range(len(dataset)), desc="Saving WebDataset"):
            # Get sample from dataset
            batch_images, batch_time_stamps, batch_cloud_masks, batch_ground_truths, temp_cal = dataset[idx]
            
            # Skip if no valid patches
            if len(batch_images) == 0:
                continue
            
            # Process each patch in the batch
            for i in range(len(batch_images)):
                
                if (batch_ground_truths[i] == 0).all():
                    continue
    
                if sample_id % samples_per_shard == 0:
                    if sink is not None:
                        sink.close()
                    shard_path = os.path.join(output_dir, f"shard-{shard_id:05d}.tar")
                    sink = wds.TarWriter(shard_path)
                    shard_id += 1

                # Create the sample dict
                sample = {
                    "__key__": f"{sample_id:08d}",
                    "image.npy": batch_images[i].astype(np.float32),
                    "timestamps.npy": batch_time_stamps[i].astype(np.int16),
                    "cloud_mask.npy": batch_cloud_masks[i].astype(np.int16),
                    "ground_truth.npy": batch_ground_truths[i].astype(np.int16),
                    "temp_cal.npy": temp_cal.astype(np.float32),
                }

                sink.write(sample)
                sample_id += 1

                if max_samples is not None and sample_id >= max_samples:
                    break

            if max_samples is not None and sample_id >= max_samples:
                break

    finally:
        if sink is not None:
            sink.close()

    print(f"Saved {sample_id} samples in {shard_id} shards to {output_dir}")
    
if __name__ == "__main__":
    
    # Hardcoded config structure (replacing read_yaml)
    config = {
        'DATASETS': {
            'train': {'dataset': 'Sentinel_fold1'},
            'eval': {'dataset': 'Sentinel_fold1'}, 
            'test': {'dataset': 'Sentinel_fold1'}
        }
    }
    
    # Hardcoded dataset info (replacing read_yaml("data/datasets.yaml"))
    DATASET_INFO = {
        'Sentinel_fold1': {
            'basedir': '/srv/data',
            'crop_train': 'Sentinel_2022',
            'crop_eval': 'Sentinel_2023', 
            'crop_test': 'Sentinel_2021',
            'gt_train': 'Crop_GTs_2022',
            'gt_eval': 'Crop_GTs_2023',
            'gt_test': 'Crop_GTs_2021',
            'temp_train': 'Temp_Calendar_2022',
            'temp_eval': 'Temp_Calendar_2023',
            'temp_test': 'Temp_Calendar_2021',
            'crop_map': 'crop_mappings.csv'
        }
    }
    
    train_config = config['DATASETS']['train']
    eval_config  = config['DATASETS']['eval']
    test_config = config['DATASETS']['test']
    datasets = {}
    
    # TRAIN data -------------------------------------------------------------------------------------------------------
    train_config['base_dir'] = DATASET_INFO[train_config['dataset']]['basedir']
    train_config['crop_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['crop_train'])
    train_config['gt_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['gt_train'])
    train_config['temp_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['temp_train'])
    train_config['crop_map'] = DATASET_INFO[train_config['dataset']]['crop_map']
    
    datasets['train'] = PreprocessingDataset(
        sentinel2_dir=train_config['crop_paths'], 
        gt_dir=train_config['gt_paths'],
        temp_calendar_dir=train_config['temp_paths'],
        label_sheet_file=train_config['crop_map'],
        patch_size=16
    )
        
    # EVAL data --------------------------------------------------------------------------------------------------------
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    eval_config['crop_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['crop_eval'])
    eval_config['gt_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['gt_eval'])
    eval_config['temp_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['temp_eval'])
    eval_config['crop_map'] = DATASET_INFO[eval_config['dataset']]['crop_map']
    
    datasets['eval'] = PreprocessingDataset(
        sentinel2_dir=eval_config['crop_paths'], 
        gt_dir=eval_config['gt_paths'],
        temp_calendar_dir=eval_config['temp_paths'],
        label_sheet_file=eval_config['crop_map'],
        patch_size=16
    )
        
    # TEST data --------------------------------------------------------------------------------------------------------
    test_config['base_dir'] = DATASET_INFO[test_config['dataset']]['basedir']
    test_config['crop_paths'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['crop_test'])
    test_config['gt_paths'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['gt_test'])
    test_config['temp_paths'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['temp_test'])
    test_config['crop_map'] = DATASET_INFO[test_config['dataset']]['crop_map']
    
    datasets['test'] = PreprocessingDataset(
        sentinel2_dir=test_config['crop_paths'], 
        gt_dir=test_config['gt_paths'],
        temp_calendar_dir=test_config['temp_paths'],
        label_sheet_file=test_config['crop_map'],
        patch_size=16
    )
    
    ####################################################
    
    save_to_webdataset(
        dataset=datasets['train'],
        output_dir="datasets/CropFormer2022_filtered",
        samples_per_shard=10_000,
        max_samples=5_000_000
    )
    
    save_to_webdataset(
        dataset=datasets['eval'],
        output_dir="datasets/CropFormer2023_filtered",
        samples_per_shard=10_000,
        max_samples=5_000_000
    )
    
    save_to_webdataset(
        dataset=datasets['test'],
        output_dir="datasets/CropFormer2021_filtered",
        samples_per_shard=10000,
        max_samples=5_000_000
    )