from __future__ import print_function, division
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms

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

def Sentinel_transform(
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
    transform_list.append(Normalize(CHANNEL_MEANS, CHANNEL_STDS))
    transform_list.append(TileDates(timestamp_mode))  # Add time_stamps as an additional channel
    transform_list.append(RandomRotate())
    transform_list.append(UnkMask())    # Generate unknown masks with numpy arrays
    transform_list.append(RandomCrop(img_res, is_training, select_k_patches))  # Updated parameter name
    
    # No ToTensor class - we'll handle tensor conversion in the collate function
    return transforms.Compose(transform_list)


def Webdataset_transforms(sample_tuple, temp_length, truncate_month, timestamp_mode, condition='open_sky'):
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
                
        if len(sel) < self.temporal_length:
            miss = self.temporal_length - len(sel)
            available_indices = [i for i in range(len(time_stamps)) if i not in sel]
            
            if len(available_indices) > 0:
                # If we have enough available indices, sample without replacement
                if len(available_indices) >= miss:
                    extra = np.random.choice(available_indices, miss, replace=False)
                else:
                    # If we don't have enough, take all available and sample the rest with replacement
                    extra = available_indices + list(np.random.choice(available_indices, miss - len(available_indices), replace=True))
                sel.extend(extra)
            else:
                # If no available indices, sample with replacement from existing sel
                extra = np.random.choice(sel, miss, replace=True)
                sel.extend(extra)
            
        sel = sorted(sel[:num_seg])
        new_time_stamps = np.array(time_stamps[sel])
        
        if self.use_temperature_calendar:
            # Convert to temperature calendar time
            new_time_stamps = temp_cal[new_time_stamps.tolist()].astype(int)
            
        return images[:, sel, ...], new_time_stamps, cloud_mask[sel], ground_truth
    
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
            extra = np.random.choice([i for i in range(len(time_stamps)) if i not in sel], miss, replace=False)
            sel.extend(extra)
        sel = sorted(sel)
        
        images = images[:, sel, ...]
        temp_cal = temp_cal[sel]
        cloud_mask = cloud_mask[sel]
        
        return images, temp_cal, cloud_mask, ground_truth
    
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


class UnkMask(object):
    """
    Create an unknown mask from cloud_mask
    """
    def __call__(self, sample):
        images, cloud_mask, ground_truth = sample
        
        ####### whole year ###################################################
        # aggregation_mapping = {
        #     # 0. Background / Non-crop
        #     0: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0,
        #     # 1. Meadow / Pasture
        #     36: 1, 37: 1,
        #     # 2. Soft Winter Wheat
        #     7: 2, 8: 2, 9: 2,
        #     # 3. Corn (Maize)
        #     6: 3,
        #     # 4. Winter Barley
        #     2: 4,
        #     # 5. Winter Rapeseed
        #     16: 5,
        #     # 6. Spring Barley
        #     1: 6,
        #     # 7. Sunflower
        #     18: 7,
        #     # 8. Grapevine
        #     39: 8, 43: 8,
        #     # 9. Beet
        #     12: 9, 13: 9,
        #     # 10. Winter Triticale
        #     4: 10,
        #     # 12. Fruits, Vegetables, Flowers
        #     24: 11, 25: 11, 27: 11, 28: 11,
        #     # 13. Potatoes
        #     14: 12,
        #     # 14. Leguminous Fodder
        #     21: 13, 22: 13, 23: 13, 34: 13, 38: 13,
        #     # 15. Soybeans
        #     17: 14,
        #     # 16. Orchard
        #     40: 15, 41: 15, 42: 15, 44: 15, 45: 15,
        #     # 17. Berries
        #     30: 16,
        #     # 18. Mixed Cereal
        #     3: 17, 5: 17, 10: 17, 11: 17, 29: 17,
        #     # 19. Sorghum
        #     26: 18,
        #     # 20. Other Oilseeds
        #     15: 19, 19: 19, 20: 19, 35: 19,
        #     # 21. Special Non-field / Perennial
        #     32: 20, 33: 20,
        #     # 22. Void label
        #     31: 21,
        # }
        ####### whole year ###################################################
        
        ####### midseson classification ###################################################
        # aggregation_mapping = {
        #     # 0. Background / Non-crop  YES
        #     0: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0,
        #     # 1. Meadow / Pasture  YES
        #     36: 1, 37: 1,
        #     # 2. Soft Winter Wheat YES
        #     7: 2, 8: 2, 9: 2,
        #     # 3. Corn (Maize) YES
        #     6: 3,
        #     # 4. Winter Barley YES
        #     2: 4,
        #     # 5. Winter Rapeseed  YES
        #     16: 5,
        #     # 6. Spring Barley  NO
        #     1: 12,
        #     # 7. Sunflower  NO
        #     18: 12,
        #     # 8. Vines  YES
        #     39: 6, 43: 6,
        #     # 9. Beet  YES
        #     12: 7, 13: 7,
        #     # 10. Winter Triticale  NO
        #     4: 12,
        #     # 12. Fruits, Vegetables, Flowers  NO
        #     24: 12, 25: 12, 27: 12, 28: 12,
        #     # 13. Potatoes  YES
        #     14: 8,
        #     # 14. Leguminous Fodder  NO
        #     21: 12, 22: 12, 23: 12, 34: 12, 38: 12,
        #     # 15. Soybeans  NO
        #     17: 12,
        #     # 16. Orchard  YES
        #     40: 9, 41: 9, 42: 9, 44: 9, 45: 9,
        #     # 17. Berries  YES
        #     30: 10,
        #     # 18. Mixed Cereal  ONLY 10, 11
        #     3: 12, 5: 12, 10: 11, 11: 11, 29: 12,
        #     # 19. Sorghum  NO
        #     26: 12,
        #     # 20. Other Oilseeds  NO
        #     15: 12, 19: 12, 20: 12, 35: 12,
        #     # 21. Special Non-field / Perennial  NO
        #     32: 12, 33: 12,
        #     # 22. Void label
        #     31: 12,
        # }
        ####### midseson classification ###################################################
        
        ####### mapping logic #############################################################
        # def map_indices(input_array):
        #     # Create a vectorized mapping function
        #     vectorized_map = np.vectorize(lambda x: aggregation_mapping.get(x, x))
        #     return vectorized_map(input_array)
        
        # ground_truth = map_indices(ground_truth)
        ####### mapping logic #############################################################
        
        # Create unknown mask based on void class
        unk_masks = (ground_truth != 31).astype(bool)
        
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