from __future__ import print_function, division
import os
import torch
import glob
import io
import pandas as pd
import numpy as np
import webdataset as wds
from data.Sentinel.data_transforms import (
    TruncateTimeDimension, SlidingWindowSubsample, AdaptiveTemperatureSubsampling,
    NoSlidingSubsample, TemperatureCalendarNoSlidingSubsample, TileDates,
    RandomRotate, UnkMask, Normalize
)
import warnings
warnings.filterwarnings("ignore")


# Global channel statistics
CHANNEL_STATS = {
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

BANDS = ['s2_B02', 's2_B03', 's2_B04', 's2_B08', 's2_B05', 's2_B06', 's2_B07', 's2_B8A', 's2_B12']
CHANNEL_MEANS = np.array([CHANNEL_STATS[band]['mean'] for band in BANDS], dtype=np.float32)
CHANNEL_STDS = np.array([CHANNEL_STATS[band]['std'] for band in BANDS], dtype=np.float32)

def npy_loads(data):
    """Load data from npy format. Imports numpy only if necessary."""
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)

def transforms(sample_tuple, temp_length, truncate_month, timestamp_mode, condition='open_sky'):
    """
    Apply the transform pipeline to a sample tuple
    """
    temporal_length = int(temp_length * (truncate_month / 12.0))
    
    transform_list = []
    transform_list.append(TruncateTimeDimension(truncate_month))
    
    # Different subsampling methods
    if timestamp_mode == 'base':
        transform_list.append(NoSlidingSubsample(temporal_length))
    elif timestamp_mode == 'sliding_window':
        transform_list.append(SlidingWindowSubsample(temporal_length, condition, use_temperature_calendar=False))
    elif timestamp_mode == 'temp_no_sw':
        transform_list.append(TemperatureCalendarNoSlidingSubsample(temporal_length))
    elif timestamp_mode == 'temp_sw':
        transform_list.append(SlidingWindowSubsample(temporal_length, condition, use_temperature_calendar=True))
    elif timestamp_mode == 'temp_subsample':
        transform_list.append(AdaptiveTemperatureSubsampling(temporal_length, condition))
    else:
        raise ValueError(f"Unknown timestamp mode: {timestamp_mode}")
       
    transform_list.append(Normalize(CHANNEL_MEANS, CHANNEL_STDS))
    transform_list.append(TileDates(timestamp_mode))
    transform_list.append(RandomRotate())
    transform_list.append(UnkMask())
    
    # Apply transforms sequentially
    for transform in transform_list:
        sample_tuple = transform(sample_tuple)
    
    return sample_tuple


class WebDatasetTransform:
    def __init__(self, temp_length, truncate_month, timestamp_mode, condition='open_sky'):
        self.temp_length = temp_length
        self.truncate_month = truncate_month
        self.timestamp_mode = timestamp_mode
        self.condition = condition

    def __call__(self, sample):
        # Extract data
        images = npy_loads(sample["image.npy"])
        time_stamps = npy_loads(sample["timestamps.npy"])
        cloud_mask  = npy_loads(sample["cloud_mask.npy"])
        ground_truth= npy_loads(sample["ground_truth.npy"])
        temp_cal    = npy_loads(sample["temp_cal.npy"])
        
        # apply transforms()
        final_images, final_unk_masks, final_ground_truth = transforms(
            (images, time_stamps, cloud_mask, temp_cal, ground_truth),
            self.temp_length,
            self.truncate_month,
            self.timestamp_mode,
            self.condition,
        )
        
        final_images = np.ascontiguousarray(final_images)
        final_unk_masks = np.ascontiguousarray(final_unk_masks)
        final_ground_truth = np.ascontiguousarray(final_ground_truth)
        
        return {
            'inputs':     torch.from_numpy(final_images).float(),
            'labels':     torch.from_numpy(final_ground_truth).int(),
            'unk_masks':  torch.from_numpy(final_unk_masks).bool(),
        }


def get_dataloader(webdataset_dir, temp_length, truncate_month, timestamp_mode, 
                   is_training, batch_size=32, num_workers=4, shuffle=True, 
                   condition='open_sky'):
    """
    Create WebDataset dataloader for preprocessed data
    """
    # Get transform function
    transform_fn = WebDatasetTransform(
        temp_length, truncate_month, timestamp_mode, condition
    )
    
    shards = glob.glob(os.path.join(webdataset_dir, "*.tar"))
    
    if is_training:
        dataset = (
            wds.WebDataset(shards, resampled=True, shardshuffle=True, workersplitter= wds.split_by_worker)
            .shuffle(1000)  # Sample-level shuffle
            .map(transform_fn)
            .batched(batch_size)
        )
        dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    
        # Unbatch, shuffle between workers, then rebatch for better mixing
        # if num_workers > 1:
        #     dataloader = dataloader.unbatched().shuffle(10).batched(batch_size)
        
        # Set epoch size (adjust based on your dataset size)
        dataloader = dataloader.with_epoch(3200 // batch_size).with_length(3200 // batch_size)
    else:
        dataset = (
            wds.WebDataset(shards, shardshuffle=False, resampled=False, workersplitter= wds.split_by_worker)
            .map(transform_fn)
            .batched(batch_size)
        )
        
        dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)

    return dataloader


def get_distr_dataloader(webdataset_dir, temp_length, truncate_month, timestamp_mode,
                        is_training, world_size, rank, batch_size=32, num_workers=4,
                        shuffle=False, condition='open_sky'):
    """
    Create distributed WebDataset dataloader for preprocessed data
    """
    # Get transform function
    transform_fn = WebDatasetTransform(
        temp_length, truncate_month, timestamp_mode, condition
    )
    
    shards = glob.glob(os.path.join(webdataset_dir, "*.tar"))
    
    if is_training:
        dataset = (
            wds.WebDataset(shards, resampled=True, shardshuffle=True, nodesplitter=wds.split_by_node, workersplitter= wds.split_by_worker)
            .shuffle(1000)  # Sample-level shuffle
            .map(transform_fn)
            .batched(batch_size)
        )
        dataloader = wds.WebLoader(
            dataset, 
            num_workers=num_workers, 
            batch_size=None,
            pin_memory=True
        )
    
        # Unbatch, shuffle between workers, then rebatch for better mixing
        # dataloader = dataloader.unbatched().shuffle(1000).batched(batch_size)
    
        # Set epoch size based on estimated samples per world_size
        dataloader = dataloader.with_epoch(1_300_000 // world_size)
        
    else:
        dataset = (
            wds.WebDataset(shards, shardshuffle=False, resampled=False, nodesplitter=wds.split_by_node, workersplitter= wds.split_by_worker)
            .map(transform_fn)
            .batched(batch_size)
        )
        
        dataloader = wds.WebLoader(
            dataset, 
            num_workers=num_workers, 
            batch_size=None,
            pin_memory=True
        )
    
    return dataloader


def get_class_weights(pixel_count_file, label_sheet_file, beta=0.9):
    """
    Calculate class weights from pixel counts (standalone function)
    """
    # Load label mapping
    label_sheet = pd.read_csv(label_sheet_file)
    csv_key = '4th_tier_ENG'
    mapping_dict = {}
    unique_codes = label_sheet[csv_key].unique()
    for idx, code in enumerate(unique_codes):
        mapping_dict[code] = idx + 1
    
    target_mapping = {0: 0, -1: 0}
    for _, row in label_sheet.iterrows():
        target_mapping[int(row['LNF_code'])] = mapping_dict[row[csv_key]]
    
    num_classes = len(unique_codes) + 1
    
    # Load pixel counts
    pixel_counts = {}
    with open(pixel_count_file, 'r') as f:
        for line in f:
            if line.strip():
                lnf_code, count = map(int, line.split(','))
                pixel_counts[lnf_code] = count
    
    # Calculate class weights
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for lnf_code, count in pixel_counts.items():
        if lnf_code in target_mapping:
            class_id = target_mapping[lnf_code]
            class_counts[class_id] += count
    
    missing = class_counts == 0
    effective_num = 1.0 - np.power(beta, class_counts)
    effective_num[effective_num <= 0] = 1e-6
    weights = (1.0 - beta) / effective_num
    weights[missing] = 1e-6
    weights = weights / np.sum(weights) * num_classes
    
    return weights
