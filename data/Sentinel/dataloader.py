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
from itertools import chain
import warnings
warnings.filterwarnings("ignore")

def my_collate(batch):
    """
    Efficient collate function for fixed-length image/mask/label lists per sample.
    Each sample is a tuple: (images_list, unk_masks_list, ground_truth_list),
    where each list contains the same number of arrays across all samples.
    """
    # Flatten all lists across the batch
    images = list(chain.from_iterable(sample[0] for sample in batch))
    unk_masks = list(chain.from_iterable(sample[1] for sample in batch))
    labels = list(chain.from_iterable(sample[2] for sample in batch))

    # Stack all at once
    inputs = torch.from_numpy(np.stack(images)).float()
    unk_masks = torch.from_numpy(np.stack(unk_masks))[..., None]
    labels = torch.from_numpy(np.stack(labels)).long()[..., None]

    return {
        'inputs': inputs,
        'labels': labels,
        'unk_masks': unk_masks
    }

def get_dataloader(crop_path, gt_path, temp_path, crop_map, temp_length, truncate_month, timestamp_mode, img_res,
                            is_training, batch_size=32, num_workers=4, shuffle=True):
    
    dataset = Sentinel2Dataset(crop_path, gt_path, temp_path, label_sheet_file=crop_map, temporal_length=temp_length, img_res=img_res,
                               truncate_month=truncate_month, timestamp_mode=timestamp_mode, is_training=is_training)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader


def get_distr_dataloader(crop_path, gt_path, temp_path, crop_map, temp_length, truncate_month, timestamp_mode, img_res,
                            is_training, world_size, rank, batch_size=32, num_workers=4, shuffle=False):
    """
    return a distributed dataloader
    """
    
    dataset = Sentinel2Dataset(crop_path, gt_path, temp_path, label_sheet_file=crop_map, temporal_length=temp_length, img_res=img_res,
                               truncate_month=truncate_month, timestamp_mode=timestamp_mode, is_training=is_training)
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
        truncate_month=12,  # Month time dimension to keep 
        img_res = 24,
        is_training=True
    ):
        self.sentinel2_dir = sentinel2_dir
        self.label_sheet_file = label_sheet_file
        self.gt_dir = gt_dir
        self.temp_calendar_dir = temp_calendar_dir
        self.bands = bands
        self.truncate_month = truncate_month
        self.temporal_length = int(temporal_length * (self.truncate_month / 12.0))  # Adjust temporal length based on truncate_month
        self.condition = condition
        self.timestamp_mode = timestamp_mode
        self.truncate_month = truncate_month
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

        self.map_lnf_code_to_ground_truth()
        self.transform = Sentinel_transform(
            temporal_length=self.temporal_length,
            truncate_month=self.truncate_month,
            condition=self.condition,
            timestamp_mode=self.timestamp_mode,
            img_res=self.img_res,
            is_training=self.is_training
        )

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
