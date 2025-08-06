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
    truncate_month=12, 
    condition='open_sky',
    timestamp_mode='base',
    img_res=24,  # Default image resolution
    is_training=True,
    select_k_patches=8  # Number of patches to select during training
):
    """
    Create a transform pipeline for Sentinel-2 data
    """
    transform_list = []
    
    # Add transformations based on parameters
    transform_list.append(RemoveDuplicateTimestamps())
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
       
    
    # Final transformations
    transform_list.append(Normalize(channel_means, channel_stds))
    transform_list.append(TileDates(timestamp_mode))  # Add time_stamps as an additional channel
    transform_list.append(RandomRotate())
    transform_list.append(UnkMask())    # Generate unknown masks with numpy arrays
    transform_list.append(RandomCrop(img_res, is_training, select_k_patches))  # Updated parameter name
    
    # No ToTensor class - we'll handle tensor conversion in the collate function
    
    return transforms.Compose(transform_list)

class TileDates(object):
    """
    Tile the time_stamps to H×W dimensions and concatenate as an additional channel.
    After Normalize(), images have shape T×H×W×C
    """
    def __init__(self, cropping_mode='random'):
        self.cropping_mode = cropping_mode
    def __call__(self, sample):
        images, time_stamps, cloud_mask, ground_truth = sample
        
        # Normalize time_stamps
        if self.cropping_mode == 'base' or self.cropping_mode == 'sliding_window':
            # We have day of year data
            time_stamps = (time_stamps - 1) / 365.0  # Normalize to [0, 1] range
        else:
            # We have temperature calendar data: multiple methods but we use a cube square root
            time_stamps = np.cbrt(time_stamps)
        
        # After Normalize(), images already have shape T×H×W×C
        T, H, W, C = images.shape
        
        # Reshape time_stamps to match images dimensions (T×H×W×1)
        tiled_timestamps = np.tile(time_stamps[:, np.newaxis, np.newaxis, np.newaxis], (1, H, W, 1))
        
        # Concatenate time_stamps as an additional channel
        images_with_time = np.concatenate([images, tiled_timestamps], axis=3)
        
        return images_with_time, cloud_mask, ground_truth

class UnkMask(object):
    """
    Create an unknown mask from cloud_mask
    """
    def __call__(self, sample):
        images, cloud_mask, ground_truth = sample
        
        ######## class aggregation ###########################################
        # aggregation_mapping = {
        #     # Meadow + Pasture
        #     36: 1, 37: 1,
        #     # Soft winter wheat (WinterWheat, Wheat, Spelt, Rye, EinkornWheat)
        #     9: 2, 4: 2, 11: 2, 10: 2, 7: 2,
        #     6: 3,   # Corn (Maize)
        #     2: 4,   # Winter barley
        #     16: 5,  # Winter rapeseed
        #     1: 6,   # Spring barley (SummerBarley)
        #     18: 7,  # Sunflower
        #     39: 8,  # Grapevine
        #     12: 9, 13: 9,  # Beet (Sugar_beets, Beets)
        #     27: 10, 24: 10, 25: 10, 28: 10, 30: 10, 46: 10,  # Fruits, vegetables, flowers
        #     14: 11,  # Potatoes
        #     21: 12, 22: 12, 23: 12, 38: 12,  # Leguminous fodder
        #     17: 13,  # Soybeans
        #     40: 14, 41: 14, 42: 14, 44: 14, 45: 14,  # Orchard
        #     34: 15, 5: 15, 3: 15, 29: 15,  # Mixed cereal
        #     26: 16,  # Sorghum
        #     8: 17,   # SummerWheat (standalone)
        #     15: 18,  # SummerRapeseed (standalone)
        #     19: 19,  # Linen (standalone)
        #     20: 20,  # Hemp (standalone)
        # }

        # delete_indices = {
        #     31, 32, 33, 35, 43, 47, 48, 49, 50, 51, 52
        # }
        
        # def map_indices(input_array):
        #     # Create a vectorized mapping function
        #     vectorized_map = np.vectorize(lambda x: aggregation_mapping.get(x, x))
        #     return vectorized_map(input_array)
        
        # ground_truth = map_indices(ground_truth)
        # unk_masks = ~np.isin(ground_truth, list(delete_indices))
        ######################################################################
        
        # Create unknown mask based on leave out class
        # unk_masks = (ground_truth != 31).astype(bool)
        
        # Create unknown mask when all classes are considered
        # Extract height and width from ground_truth
        H, W = ground_truth.shape[0], ground_truth.shape[1]
        unk_masks = np.ones((H, W), dtype=np.bool_)
        
        return images, unk_masks, ground_truth

class RandomCrop(object):
    """
    Crop patches of size img_res × img_res from the input image.
    """
    def __init__(self, img_res=24, is_training=True, select_k_patches=1):
        self.img_res = img_res
        self.is_training = is_training
        self.select_k_patches = select_k_patches  # Number of patches to select during training
        
    def __call__(self, sample):
        images, unk_masks, ground_truth = sample
        
        # Get the original dimensions
        T, H, W, C = images.shape
        
        # Calculate padding needed to make H (and W) a multiple of img_res
        pad_size = (self.img_res - H % self.img_res) % self.img_res
        
        # Zero pad if needed
        if pad_size > 0:
            images = np.pad(images, ((0, 0), (0, pad_size), (0, pad_size), (0, 0)), mode='constant', constant_values=0)
            # Pad unk_masks with 1s
            unk_masks = np.pad(unk_masks, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=1)
            # Pad ground_truth with 0s
            ground_truth = np.pad(ground_truth, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
        
        # Update dimensions after padding
        T, H, W, C = images.shape
        
        # Calculate how many patches we can get in each dimension
        num_patches = H // self.img_res
        
        if self.is_training:
            # Training: randomly select k patches
            total_patches = num_patches * num_patches
            k_patches = min(self.select_k_patches, total_patches)  # Don't select more patches than available
            
            # Generate all possible patch positions
            all_patches = []
            for h in range(num_patches):
                for w in range(num_patches):
                    all_patches.append((h * self.img_res, w * self.img_res))
            
            # Randomly select k patches without replacement
            selected_patches = random.sample(all_patches, k_patches)
        else:
            # Evaluation: return all patches
            selected_patches = []
            for h in range(num_patches):
                for w in range(num_patches):
                    selected_patches.append((h * self.img_res, w * self.img_res))
        
        # Extract the selected patches
        cropped_images = []
        cropped_unk_masks = []
        cropped_ground_truths = []
        
        for h_start, w_start in selected_patches:
            # Extract patches
            patch_img = images[:, h_start:h_start+self.img_res, w_start:w_start+self.img_res, :]
            patch_unk_mask = unk_masks[h_start:h_start+self.img_res, w_start:w_start+self.img_res]
            patch_gt = ground_truth[h_start:h_start+self.img_res, w_start:w_start+self.img_res]
            
            cropped_images.append(patch_img)
            cropped_unk_masks.append(patch_unk_mask)
            cropped_ground_truths.append(patch_gt)
        
        return cropped_images, cropped_unk_masks, cropped_ground_truths

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
        images, cloud_mask, ground_truth = sample
        
        # Augmentation
        if random.random() < self.probability:
            k = random.randint(1, 3)
            images = np.rot90(images, k=k, axes=(1, 2))
            if ground_truth.ndim == 2:
                ground_truth = np.rot90(ground_truth, k=k, axes=(0, 1))
                
        return images, cloud_mask, ground_truth

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
    Truncate the time dimension based on the truncate_month parameter
    """
    def __init__(self, truncate_month=12):
        self.truncate_month = truncate_month
        
    def __call__(self, sample):
        images, time_stamps, cloud_mask, temp_cal, ground_truth = sample
        
        if self.truncate_month < 12:
            # Convert day of year to month (approximate)
            days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            cumulative_days = np.cumsum([0] + days_per_month)
            
            # Find indices where timestamps fall within the specified month range
            valid_indices = []
            for i, day in enumerate(time_stamps):
                month = np.searchsorted(cumulative_days, day, side='right')
                if month <= self.truncate_month:
                    valid_indices.append(i)
            
            if valid_indices:
                valid_indices = np.array(valid_indices)
                images = images[:, valid_indices, ...]
                time_stamps = time_stamps[valid_indices]
                cloud_mask = cloud_mask[valid_indices]
                
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
            # OG
            # extra = np.linspace(0, len(time_stamps) - 1, miss).astype(int)
            # sel.extend([i for i in extra if i not in sel])
            extra = np.random.choice([i for i in range(len(time_stamps)) if i not in sel], miss, replace=False)
            sel.extend(extra)
        sel = sorted(sel)
        
        images = images[:, sel, ...]
        temp_cal = temp_cal[sel]
        cloud_mask = cloud_mask[sel]
        
        return images, temp_cal, cloud_mask, ground_truth

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
