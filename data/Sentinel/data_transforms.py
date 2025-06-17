from __future__ import print_function, division
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import zarr

def Sentinel_transform(
    channel_means, 
    channel_stds, 
    temporal_length, 
    truncate_portion=1.0, 
    condition='open_sky',
    timestamp_mode='base',
    cropping_mode='random'
):
    """
    Create a transform pipeline for Sentinel-2 data
    """
    transform_list = []
    
    # Add transformations based on parameters
    transform_list.append(RemoveDuplicateTimestamps())
    transform_list.append(TruncateTimeDimension(truncate_portion))
    
    # Different subsampling methods
    if timestamp_mode == 'temp_subsample':
        transform_list.append(AdaptiveTemperatureSubsampling(temporal_length, condition))
    elif timestamp_mode == 'temp_sw':
        transform_list.append(SlidingWindowSubsample(temporal_length, condition, use_temperature_calendar=True))
    elif timestamp_mode == 'temp_no_sw':
        transform_list.append(TemperatureCalendarNoSlidingSubsample(temporal_length))
    elif timestamp_mode == 'base':
        transform_list.append(NoSlidingSubsample(temporal_length))
    else:
        transform_list.append(SlidingWindowSubsample(temporal_length, condition, use_temperature_calendar=False))
    
    # Final transformations
    transform_list.append(Normalize(channel_means, channel_stds))
    transform_list.append(TileDates())  # Add time_stamps as an additional channel
    transform_list.append(RandomRotate())
    transform_list.append(ToTensor())
    transform_list.append(UnkMask())  # Add UnkMask transformation
    transform_list.append(FormatOutput())
    
    return transforms.Compose(transform_list)

class TileDates(object):
    """
    Tile the time_stamps to H×W dimensions and concatenate as an additional channel.
    After Normalize(), images have shape T×H×W×C
    """
    def __call__(self, sample):
        images, time_stamps, cloud_mask, ground_truth = sample
        
        # After Normalize(), images already have shape T×H×W×C
        T, H, W, C = images.shape
        
        # Reshape time_stamps to match images dimensions (T×H×W×1)
        tiled_timestamps = np.tile(time_stamps[:, np.newaxis, np.newaxis, np.newaxis], (1, H, W, 1))
        
        # Concatenate time_stamps as an additional channel
        images_with_time = np.concatenate([images, tiled_timestamps], axis=3)
        
        return images_with_time, time_stamps, cloud_mask, ground_truth

class UnkMask(object):
    """
    Create an unknown mask from cloud_mask
    """
    def __call__(self, sample):
        images, time_stamps, cloud_mask, ground_truth = sample
        
        # Create unknown mask based on cloud_mask (valid = not cloudy)
        # Assuming cloud_mask == 1 means cloudy and cloud_mask == 0 means clear
        unk_masks = (cloud_mask == 0).to(torch.bool)
        
        return images, time_stamps, unk_masks, ground_truth

class FormatOutput(object):
    """
    Format final output as a dictionary for model input
    """
    def __call__(self, sample):
        images, time_stamps, unk_masks, ground_truth = sample
        
        # Create a dictionary with the proper keys
        return {
            'inputs': images,
            'time_stamps': time_stamps,
            'labels': ground_truth,
            'unk_masks': unk_masks,  # Use the generated unk_masks
            'seq_lengths': images.shape[0]
        }

class ToTensor(object):
    """
    Convert numpy arrays to torch tensors
    """
    def __call__(self, sample):
        images, time_stamps, cloud_mask, ground_truth = sample
        
        # Convert to torch tensors
        images = torch.from_numpy(images.copy())
        ground_truth = torch.from_numpy(ground_truth.copy())
        time_stamps = torch.from_numpy(time_stamps)
        cloud_mask = torch.from_numpy(cloud_mask.copy())
        
        return images, time_stamps, cloud_mask, ground_truth

class Normalize(object):
    """
    Normalize inputs based on channel statistics
    """
    def __init__(self, channel_means, channel_stds):
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, ground_truth = sample
        
        # Normalize
        images = (images - self.channel_means[:, None, None, None]) / self.channel_stds[:, None, None, None]
        images = np.transpose(images, (1, 2, 3, 0))  # T x H x W x C
        
        return images, time_stamps, cloud_mask, ground_truth

class RandomRotate(object):
    """
    Random rotation for augmentation
    """
    def __init__(self, probability=0.5):
        self.probability = probability
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, ground_truth = sample
        
        # Augmentation
        if random.random() < self.probability:
            k = random.randint(1, 3)
            images = np.rot90(images, k=k, axes=(2, 3))
            if ground_truth.ndim == 2:
                ground_truth = np.rot90(ground_truth, k=k, axes=(0, 1))
                
        return images, time_stamps, cloud_mask, ground_truth

class RemoveDuplicateTimestamps(object):
    """
    Remove duplicate timestamps from the data using numpy's unique function
    """
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
        
        # Get unique timestamps and their indices (keeping the first occurrence)
        unique_times, indices = np.unique(time_stamps, return_index=True)
        
        # Sort indices to maintain chronological order
        indices = np.sort(indices)
        
        # Extract the relevant data using the indices
        filtered_images = images[:, indices, ...]
        filtered_times = time_stamps[indices]
        filtered_cloud_mask = cloud_mask[indices]
        
        return filtered_images, filtered_times, filtered_cloud_mask, temp_cal, ground_truth
    
    # This method checks for the best quality sample for each unique timestamp
    # def __call__(self, sample):
    # images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
    
    # # Find unique timestamps
    # unique_times = np.unique(time_stamps)
    # best_indices = []
    
    # # For each unique timestamp, find the best quality sample
    # for t in unique_times:
    #     # Find all occurrences of this timestamp
    #     dup_idxs = np.where(time_stamps == t)[0]
    #     if len(dup_idxs) > 1:
    #         # Calculate quality (less None values is better)
    #         none_counts = [np.sum(images[:, idx, ...] == None) for idx in dup_idxs]
    #         best = dup_idxs[np.argmin(none_counts)]
    #     else:
    #         best = dup_idxs[0]
    #     best_indices.append(best)
    
    # # Sort indices to maintain chronological order
    # best_indices = np.sort(best_indices)
    
    # return images[:, best_indices, ...], time_stamps[best_indices], cloud_mask[best_indices], temp_cal, ground_truth

class TruncateTimeDimension(object):
    """
    Truncate the time dimension based on the truncate_portion parameter
    """
    def __init__(self, truncate_portion=1.0):
        self.truncate_portion = truncate_portion
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
        
        if self.truncate_portion < 1.0:
            total = images.shape[1]
            new_t = max(1, int(total * self.truncate_portion))
            images = images[:, :new_t, ...]
            time_stamps = time_stamps[:new_t]
            cloud_mask = cloud_mask[:new_t]
            
        return images, time_stamps, cloud_mask, temp_cal, ground_truth

class SlidingWindowSubsample(object):
    """
    Subsample time dimension using sliding window approach
    """
    def __init__(self, temporal_length, condition='open_sky', use_temperature_calendar=False):
        self.temporal_length = temporal_length
        self.condition = condition
        self.use_temperature_calendar = use_temperature_calendar
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
        
        T = len(time_stamps)
        num_seg = self.temporal_length
        seg_size = max(1, T // num_seg)
        if self.condition == 'cloud':
            cov = np.sum(cloud_mask == 1, axis=(1, 2))
        else:
            cov = np.sum(cloud_mask == 0, axis=(1, 2))
        sel = []
        for start in range(0, T, seg_size):
            end = min(start + seg_size, T)
            idxs = range(start, end)
            if len(idxs) > 0:  # Check if the range is not empty
                if self.condition == 'cloud':
                    best = min(idxs, key=lambda x: cov[x])
                else:
                    best = max(idxs, key=lambda x: cov[x])
                sel.append(best)
        sel = sorted(sel[:num_seg])
        new_time_stamps = np.array(time_stamps[sel])
        
        if self.use_temperature_calendar:
            # Convert to temperature calendar time
            new_time_stamps = temp_cal[new_time_stamps.tolist()].astype(int)
            
        return images[:, sel, ...], new_time_stamps, cloud_mask[sel], ground_truth


class AdaptiveTemperatureSubsampling(object):
    """
    Subsample time dimension using temperature calendar
    """
    def __init__(self, temporal_length, condition='open_sky'):
        self.temporal_length = temporal_length
        self.condition = condition
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
            
        temp_cal = temp_cal[time_stamps.tolist()].astype(int)
        tmin, tmax = temp_cal.min(), temp_cal.max()
        target = (tmax - tmin) / self.temporal_length
        sel = []
        cur = tmin
        while len(sel) < self.temporal_length and cur < tmax:
            nxt = cur + target
            wnd = np.where((temp_cal >= cur) & (temp_cal < nxt))[0]
            wnd = wnd[wnd < len(time_stamps)]
            if len(wnd) > 0:
                if self.condition == 'cloud':
                    cc = np.sum(cloud_mask[wnd] == 1, axis=(1, 2))
                    best = wnd[np.argmin(cc)]
                else:
                    cs = np.sum(cloud_mask[wnd] == 0, axis=(1, 2))
                    best = wnd[np.argmax(cs)]
                sel.append(best)
            cur = nxt
        while len(sel) > self.temporal_length:
            if self.condition == 'cloud':
                cc = np.sum(cloud_mask[sel] == 1, axis=(1, 2))
                worst = np.argmax(cc)
            else:
                cs = np.sum(cloud_mask[sel] == 0, axis=(1, 2))
                worst = np.argmin(cs)
            sel.pop(worst)
        if len(sel) < self.temporal_length:
            miss = self.temporal_length - len(sel)
            extra = np.linspace(0, len(time_stamps) - 1, miss).astype(int)
            sel.extend([i for i in extra if i not in sel])
        sel = sorted(sel)
        
        return images[:, sel, ...], temp_cal[sel], cloud_mask[sel], ground_truth

class NoSlidingSubsample(object):
    """
    Subsample time dimension using linear indices
    """
    def __init__(self, temporal_length):
        self.temporal_length = temporal_length
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
        
        total = images.shape[1]
        indices = np.linspace(0, total - 1, self.temporal_length, dtype=int)
        
        return images[:, indices, ...], time_stamps[indices], cloud_mask[indices], ground_truth

class TemperatureCalendarNoSlidingSubsample(object):
    """
    Subsample time dimension using temperature calendar and linear indices
    """
    def __init__(self, temporal_length):
        self.temporal_length = temporal_length
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
        
        total = images.shape[1]
        indices = np.linspace(0, total - 1, self.temporal_length, dtype=int)
        new_time_stamps = temp_cal[indices]
        
        return images[:, indices, ...], new_time_stamps, cloud_mask[indices], ground_truth

