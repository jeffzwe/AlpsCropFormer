from __future__ import print_function, division
import os
import torch
import glob
import io
import pandas as pd
import numpy as np
import numpy.lib.format
import webdataset as wds
from data.Sentinel.data_transforms import Webdataset_transforms
import warnings
warnings.filterwarnings("ignore")

def npy_loads(data):
    """Load data from npy format. Imports numpy only if necessary."""
    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)

class WebDatasetTransform:
    def __init__(self, temp_length, truncate_month, timestamp_mode, condition='open_sky', rank= -1):
        self.temp_length = temp_length
        self.truncate_month = truncate_month
        self.timestamp_mode = timestamp_mode
        self.condition = condition
        self.rank = rank  # Store rank for debugging purposes   

    def __call__(self, sample):
        # Extract data
        images = npy_loads(sample["image.npy"])
        time_stamps = npy_loads(sample["timestamps.npy"])
        cloud_mask  = npy_loads(sample["cloud_mask.npy"])
        ground_truth= npy_loads(sample["ground_truth.npy"])
        temp_cal    = npy_loads(sample["temp_cal.npy"])
        
        # apply your existing transforms()
        final_images, final_unk_masks, final_ground_truth = Webdataset_transforms(
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
            'labels':     torch.from_numpy(final_ground_truth).long(),
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
        temp_length, truncate_month, timestamp_mode, condition, rank
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
            pin_memory=True,
            prefetch_factor=2
        )
    
        # Unbatch, shuffle between workers, then rebatch for better mixing
        dataloader = dataloader.unbatched().shuffle(2000).batched(batch_size)
    
        # Set epoch size based on estimated samples per world_size
        
        # Filtered dataset
        dataloader = dataloader.with_epoch(880_000 // (batch_size * world_size)).with_length(880_000 // (batch_size * world_size))
        
        # Unfiltered dataset
        # dataloader = dataloader.with_epoch(1_300_000 // (batch_size * world_size)).with_length(1_300_000 // (batch_size * world_size))
        
    else:
           
        dataset = (
            wds.WebDataset(shards, shardshuffle=False, resampled=False, workersplitter= wds.split_by_worker, nodesplitter=wds.split_by_node)
            .map(transform_fn)
            .batched(batch_size)
        )
        
        dataloader = wds.WebLoader(
            dataset, 
            num_workers=num_workers, 
            batch_size=None,
            pin_memory=True,
            prefetch_factor=2
        )
    
    return dataloader


def get_class_weights(pixel_count_file, label_sheet_file, power=0.25):
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
    
    # Filtered background pixel count
    class_counts[0] = 83764016
    
    powered_counts = np.power(class_counts, power)
    weights = 1.0 / powered_counts
    weights[class_counts == 0] = 0.0
    
    # Normalize
    weights = weights / np.sum(weights) * num_classes
    
    return weights


def get_aggregated_class_weights(power=1.0):
    """
    Calculate class weights for aggregated classes using predefined pixel counts
    """
    # Aggregated pixel counts from your calculation
    aggregated_pixel_counts = {
        0: 92749354,   # Background / Non-crop
        1: 105975224,  # Meadow / Pasture
        2: 7075195,    # Soft Winter Wheat
        3: 6163415,    # Corn (Maize)
        4: 2640192,    # Winter Barley
        5: 2468307,    # Winter Rapeseed
        # 6: 84139,      # Spring Barley
        # 7: 517357,     # Sunflower
        6: 1220443,    # Grapevine
        7: 1589654,    # Beet
        # 10: 1629399,   # Winter Triticale
        # 11: 1332751,   # Fruits, Vegetables, Flowers
        8: 1066641,   # Potatoes
        # 13: 472791,    # Leguminous Fodder
        # 14: 285505,    # Soybeans
        9: 685029,    # Orchard
        10: 109067,    # Berries
        11: 968505,     # whole year pixel counts 1343107,   # Mixed Cereal
        # 18: 56842,     # Sorghum
        # 19: 35007,     # Other Oilseeds
        # 20: 357478,    # Special Non-field / Perennial
        # 21: 1263095,   # Void label
    }
    
    num_classes = 12
    
    # Calculate class weights
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for class_id, count in aggregated_pixel_counts.items():
        class_counts[class_id] = count
    
    powered_counts = np.power(class_counts, power)
    weights = 1.0 / powered_counts
    weights[class_counts == 0] = 0.0
    
    # Normalize
    weights = weights / np.sum(weights) * num_classes
    
    return weights
