import geopandas as gpd
import os
import numpy as np
import pickle
import datetime
import torch
import argparse
import json
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

def get_doy(date):
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, '%Y.%m.%d')
    return dt.timetuple().tm_yday


def unfold_reshape(img, HW):
    if len(img.shape) == 4:
        T, C, H, W = img.shape
        img = img.unfold(2, size=HW, step=HW).unfold(3, size=HW, step=HW)
        img = img.reshape(T, C, -1, HW, HW).permute(2, 0, 1, 3, 4)
    elif len(img.shape) == 3:
        C, H, W = img.shape
        img = img.unfold(1, size=HW, step=HW).unfold(2, size=HW, step=HW)
        img = img.reshape(C, -1, HW, HW).permute(1, 0, 2, 3)

    return img


def downsample(img, scale_factor=3, target_size=None):
    """
    Downsample satellite imagery using rasterio's resampling methods.
    
    Args:
        img: Input image of shape (T, C, H, W) for satellite images or (1, H, W) for labels
        scale_factor: Factor by which to downsample (e.g., 3 means 9 pixels become 1)
        target_size: Tuple of (height, width) for the output image
        
    Returns:
        Downsampled image with same number of dimensions
    """
    original_shape = img.shape
    original_dtype = img.dtype
    
    if len(original_shape) == 4:  # (T, C, H, W) - satellite images
        T, C, H, W = original_shape
        
        if target_size:
            new_H, new_W = target_size
        else:
            new_H, new_W = H // scale_factor, W // scale_factor
            
        result = np.zeros((T, C, new_H, new_W), dtype=original_dtype)
        
        for t in range(T):
            for c in range(C):
                data = img[t, c]
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(
                        driver='GTiff',
                        height=H,
                        width=W,
                        count=1,
                        dtype=data.dtype
                    ) as dataset:
                        dataset.write(data, 1)
                        out_image = dataset.read(
                            out_shape=(new_H, new_W),
                            resampling=Resampling.bilinear
                        )
                        result[t, c] = out_image[0]
                        
    elif len(original_shape) == 3:  # (1, H, W) - labels
        C, H, W = original_shape
        
        if target_size:
            new_H, new_W = target_size
        else:
            new_H, new_W = H // scale_factor, W // scale_factor
            
        result = np.zeros((C, new_H, new_W), dtype=original_dtype)
        
        # For labels, use nearest neighbor resampling to preserve class values
        for c in range(C):
            data = img[c]
            with rasterio.MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=H,
                    width=W,
                    count=1,
                    dtype=data.dtype
                ) as dataset:
                    dataset.write(data, 1)
                    out_image = dataset.read(
                        out_shape=(new_H, new_W),
                        resampling=Resampling.nearest
                    )
                    result[c] = out_image[0]
    
    return result


def filter_temporal_resolution(img, doy, min_days_interval=15):
    """
    Filter the temporal dimension to ensure a minimum number of days between measurements.
    
    Args:
        img: Input image of shape (T, C, H, W)
        doy: Array of day-of-year values corresponding to each time step
        min_days_interval: Minimum number of days between consecutive measurements
        
    Returns:
        Filtered image and corresponding doy array
    """
    if len(doy) <= 1:
        return img, doy  # Nothing to filter if there's only one time step
    
    # Initialize with the first date
    selected_indices = [0]
    last_selected_doy = doy[0]
    
    # Iterate through all days and select those with sufficient interval
    for i in range(1, len(doy)):
        current_doy = doy[i]
        days_diff = current_doy - last_selected_doy
        
        # Handle year boundary (December to January transition)
        if days_diff < 0:  # Crossed year boundary
            days_diff += 365  # Approximation, not accounting for leap years
            
        if days_diff >= min_days_interval:
            selected_indices.append(i)
            last_selected_doy = current_doy
    
    # Apply filter to both image and doy
    filtered_img = img[selected_indices]
    filtered_doy = doy[selected_indices]
    
    # print(f"Temporal filtering: {len(doy)} dates -> {len(filtered_doy)} dates")
    
    return filtered_img, filtered_doy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CSCL pre-training')
    parser.add_argument('--rootdir', type=str, default="",
                        help='PASTIS24 root dir')
    parser.add_argument('--savedir', type=str, default="",
                        help='where to save new data')
    parser.add_argument('--HWout', type=int, default=24,
                        help='size of extracted windows')
    args = parser.parse_args()

    rootdir = args.rootdir
    savedir = args.savedir
    HWin = 128
    HWout = args.HWout
    
    # Parse target size if provided
    target_size = (48, 48)

    meta_patch = gpd.read_file(os.path.join(rootdir, "metadata.geojson"))
           
    for i in tqdm(range(meta_patch.shape[0]), desc='Processing patches', unit='patch'):
        img = np.load(os.path.join(rootdir, 'DATA_S2/S2_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        lab = np.load(os.path.join(rootdir, 'ANNOTATIONS/TARGET_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        dates = json.loads(meta_patch['dates-S2'].iloc[i])
        doy = np.array([get_doy(d) for d in dates.values()])
        idx = np.argsort(doy)
        img = img[idx]
        doy = doy[idx]
        
        # Filter temporal resolution to ensure minimum days between measurements
        img, doy = filter_temporal_resolution(img, doy, min_days_interval=15)
        
        # Apply downsampling before unfolding - use appropriate resampling for each type
        # print(f"Before downsampling - Image shape: {img.shape}, Label shape: {lab.shape}")
        img = downsample(img, target_size=target_size)
        lab = downsample(lab, target_size=target_size)
        # print(f"After downsampling - Image shape: {img.shape}, Label shape: {lab.shape}")
        
        unfolded_images = unfold_reshape(torch.tensor(img), HWout).numpy()
        unfolded_labels = unfold_reshape(torch.tensor(lab), HWout).numpy()

        for j in range(unfolded_images.shape[0]):
            sample = {'img': unfolded_images[j], 'labels': unfolded_labels[j], 'doy': doy}

            with open(os.path.join(savedir, "%d_%d.pickle" % (meta_patch['ID_PATCH'].iloc[i], j)), "wb") as output_file:
                pickle.dump(sample, output_file)
