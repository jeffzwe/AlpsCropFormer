from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
import pickle
import random
import zarr
import numpy as np
from data.Sentinel.data_transforms import Sentinel_transform
import warnings
warnings.filterwarnings("ignore")

def my_collate(batch):
    """
    Collate function that handles the consistent list format from the dataset.
    Each sample in the batch is a tuple of (images_list, unk_masks_list, ground_truth_list)
    where each list contains one or more arrays.
    
    Args:
        batch: A list of samples returned by __getitem__
        
    Returns:
        A dictionary with batched inputs, labels, and unk_masks
    """
    
    # Initialize lists to store all elements
    all_inputs = []
    all_unk_masks = []
    all_labels = []
    
    # Process each sample in the batch
    for sample in batch:
        images_list, unk_masks_list, ground_truth_list = sample
        
        # Process each element in the lists
        for i in range(len(images_list)):
            # Convert numpy arrays to float32 tensors
            all_inputs.append(torch.from_numpy(images_list[i].copy()).float())
            all_unk_masks.append(torch.from_numpy(unk_masks_list[i].copy()).unsqueeze(-1))
            all_labels.append(torch.from_numpy(ground_truth_list[i].copy()).float().unsqueeze(-1))
    
    # Stack all elements into tensors
    return {
        'inputs': torch.stack(all_inputs),
        'labels': torch.stack(all_labels),
        'unk_masks': torch.stack(all_unk_masks)
    }

def get_dataloader(crop_path, gt_path, temp_path, crop_map, temp_length, truncate_portion, timestamp_mode, cropping_mode, img_res,
                            is_training, batch_size=32, num_workers=4, shuffle=True):
    
    if cropping_mode == 'all':
        if batch_size < 16:
            raise ValueError("For batch_sizes smaller than 16 one can only use cropping_mode == 'random'")
        batch_size = batch_size // 16
    
    dataset = Sentinel2Dataset(crop_path, gt_path, temp_path, label_sheet_file=crop_map, temporal_length=temp_length, img_res=img_res,
                               truncate_portion=truncate_portion, timestamp_mode=timestamp_mode, cropping_mode=cropping_mode, is_training=is_training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                                             collate_fn=my_collate)
    return dataloader


def get_distr_dataloader(crop_path, gt_path, temp_path, crop_map, temp_length, truncate_portion, timestamp_mode, cropping_mode, img_res,
                            is_training, world_size, rank, batch_size=32, num_workers=4, shuffle=False):
    """
    return a distributed dataloader
    """
    if cropping_mode == 'all':
        if batch_size < 16:
            raise ValueError("For batch_sizes smaller than 16 one can only use cropping_mode == 'random'")
        batch_size = batch_size // 16
        
    
    dataset = Sentinel2Dataset(crop_path, gt_path, temp_path, label_sheet_file=crop_map, temporal_length=temp_length, img_res=img_res,
                               truncate_portion=truncate_portion, timestamp_mode=timestamp_mode, cropping_mode=cropping_mode, is_training=is_training)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler, collate_fn=my_collate)
    return dataloader


class Sentinel2Dataset(Dataset):
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
        temporal_length=24,
        condition='open_sky',
        sample_percentage=1.0,
        seed=42,
        timestamp_mode='base',
        truncate_portion=1.0,  # Portion of time dimension to keep (1.0 = no truncation)
        cropping_mode='random',  # Changed from cropping_select_k to cropping_mode
        img_res = 24,
        is_training=True
    ):
        self.sentinel2_dir = sentinel2_dir
        self.label_sheet_file = label_sheet_file
        self.gt_dir = gt_dir
        self.temp_calendar_dir = temp_calendar_dir
        self.bands = bands
        self.truncate_portion = truncate_portion
        self.temporal_length = int(temporal_length * self.truncate_portion)
        self.condition = condition
        self.timestamp_mode = timestamp_mode
        self.truncate_portion = truncate_portion
        self.cropping_mode = cropping_mode
        self.img_res = img_res
        self.is_training = is_training

        if seed is not None:
            random.seed(seed)

        self.data_files = self._get_data_files()

        if sample_percentage < 1.0:
            n_samples = int(len(self.data_files) * sample_percentage)
            self.data_files = random.sample(self.data_files, n_samples)

        self.target_mapping = None
        self.num_classes = 0
        self.mapping_dict = None

        # Channel statistics
        self.channel_stats = {
            's2_B02': {'mean': 1962.10, 'std': 5731.00},
            's2_B03': {'mean': 2106.59, 'std': 5656.44},
            's2_B04': {'mean': 2028.38, 'std': 5662.33},
            's2_B08': {'mean': 3797.36, 'std': 5362.84},
            's2_B05': {'mean': 2430.80, 'std': 5590.77},
            's2_B06': {'mean': 3380.80, 'std': 5408.80},
            's2_B07': {'mean': 3681.89, 'std': 5354.31},
            's2_B8A': {'mean': 3851.88, 'std': 5307.00},
            's2_B12': {'mean': 1582.97, 'std': 5161.84}
        }
        self.channel_means = np.array([self.channel_stats[band]['mean'] for band in self.bands], dtype=np.float32)
        self.channel_stds = np.array([self.channel_stats[band]['std'] for band in self.bands], dtype=np.float32)

        self.map_lnf_code_to_ground_truth()
        self.transform = Sentinel_transform(
            channel_means=self.channel_means,
            channel_stds=self.channel_stds,
            temporal_length=self.temporal_length,
            truncate_portion=self.truncate_portion,
            condition=self.condition,
            timestamp_mode=self.timestamp_mode,
            cropping_mode=self.cropping_mode,
            img_res=self.img_res,
            is_training=self.is_training
        )

        if is_training:
            print(f"Number of classes: {self.num_classes}")
            print(f"Temporal length: {self.temporal_length}")
            print(f"Number of S2 bands: {len(self.bands)}")
            print(f"Bands: {self.bands}")
            print(f"Dataset size: {len(self.data_files)}")
            # print(f"LNF code mapping: {self.target_mapping}")
            # print(f"Channel Means: {[f'{mean:.2f}' for mean in self.channel_means]}")
            # print(f"Channel Stds: {[f'{std:.2f}' for std in self.channel_stds]}")
            print(f"timestamp_mode: {self.timestamp_mode}")
            print(f"truncate_portion: {self.truncate_portion}")

    def _get_data_files(self):
        data_files = []
        # for gt_dir in self.gt_dirs:
        gt_files = [f for f in os.listdir(self.gt_dir) if f.endswith('.zarr')]
        for gt_file in gt_files:
            data_file_path = os.path.join(self.sentinel2_dir, gt_file)
            temp_calendar_path = os.path.join(self.temp_calendar_dir, gt_file)
            if os.path.exists(data_file_path) and os.path.exists(temp_calendar_path):
                data_files.append((data_file_path, os.path.join(self.gt_dir, gt_file), temp_calendar_path))
        return data_files

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
        self.num_classes = len(unique_codes) + 1
        self.mapping_dict = mapping_dict

    def get_class_weights(self, pixel_count_file, beta=0.9):
        pixel_counts = {}
        with open(pixel_count_file, 'r') as f:
            for line in f:
                if line.strip():
                    lnf_code, count = map(int, line.split(','))
                    pixel_counts[lnf_code] = count
        class_counts = np.zeros(self.num_classes, dtype=np.float32)
        for lnf_code, count in pixel_counts.items():
            if lnf_code in self.target_mapping:
                class_id = self.target_mapping[lnf_code]
                class_counts[class_id] += count
        missing = class_counts == 0
        effective_num = 1.0 - np.power(beta, class_counts)
        effective_num[effective_num <= 0] = 1e-6
        weights = (1.0 - beta) / effective_num
        weights[missing] = 1e-6
        weights = weights / np.sum(weights) * self.num_classes
        return weights

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file, gt_file, temp_calendar_file = self.data_files[idx]
        with zarr.open(data_file, mode='r') as s2_data, \
             zarr.open(gt_file, mode='r') as gt_data:

            # Load S2 and metadata
            bands_data = [s2_data[band][:].astype(np.float32) for band in self.bands]
            images = np.stack(bands_data, axis=0)  # C x T x H x W
            time_stamps = s2_data['/time'][:]
            cloud_mask = s2_data['s2_mask'][:].astype(np.int16)
            
            # Print unique values in cloud_mask
            # print(f"Sample {idx} cloud_mask unique values: {np.unique(cloud_mask)}")
            
            # Check if time dimension is sufficient after truncation
            time_dim_length = images.shape[1]
            truncated_length = int(time_dim_length * self.truncate_portion)
            if truncated_length < self.temporal_length:
                # Option 1: Raise an exception
                raise ValueError(f"Time dimension too short after truncation: got {truncated_length}, need {self.temporal_length}. "
                                f"Sample {idx}, file {data_file}. Consider using a smaller temporal_length or larger truncate_portion.")
        
        # Load GT
        lnf = gt_data['lnf_code'][:]
        lnf = np.where(lnf == None, 0, lnf).astype(np.int32)
        ground_truth = np.vectorize(lambda x: self.target_mapping.get(x, 0))(lnf).astype(np.int32)
    
        # Load temperature calendar
        with zarr.open(temp_calendar_file, mode='r') as temp_data:
            temp_cal = temp_data['temperature_calendar'][:].astype(np.float32)
        
        # Create sample
        sample = (images, time_stamps, cloud_mask, temp_cal, ground_truth)
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        return sample
