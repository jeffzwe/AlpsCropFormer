import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device
from data import get_dataloaders
from tqdm import tqdm

# Set seaborn style for publication-ready plots
sns.set_style("whitegrid")
sns.set_palette("viridis")

if __name__ == "__main__":
    
    
    ######## PASTIS ##########################################################
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config_file', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str,
                         help='gpu ids to use')

    args = parser.parse_args()
    config_file = args.config_file
    device_ids = [int(d) for d in args.device.split(',')]

    device = get_device(device_ids, allow_cpu=True)  # Allow CPU for apple silicon compatibility

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)
    
    # Define class names mapping
    class_names = {
        0: 'Background',
        1: 'Meadow',
        2: 'Soft winter wheat',
        3: 'Corn',
        4: 'Winter barley',
        5: 'Winter rapeseed',
        6: 'Spring barley',
        7: 'Sunflower',
        8: 'Grapevine',
        9: 'Beet',
        10: 'Winter triticale',
        11: 'Winter durum wheat',
        12: 'Fruits, vegetables, flowers',
        13: 'Potatoes',
        14: 'Leguminous fodder',
        15: 'Soybeans',
        16: 'Orchard',
        17: 'Mixed cereal',
        18: 'Sorghum',
        19: 'Void label'
    }
    
    # Get the training dataset
    train_dataset = dataloaders['train'].dataset
    
    # Initialize pixel counts dictionary
    pixel_counts = {}
    
    # Iterate through the entire dataset
    print("Counting pixels per class...")
    for i in tqdm(range(len(train_dataset))):
        sample = train_dataset[i]
        labels = sample['labels']
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(labels):
            labels = labels.numpy()
        
        # Count unique values and their frequencies
        unique_classes, counts = np.unique(labels, return_counts=True)
        
        # Accumulate counts
        for class_id, count in zip(unique_classes, counts):
            if class_id in pixel_counts:
                pixel_counts[class_id] += count
            else:
                pixel_counts[class_id] = count
    
    # Sort by pixel count in descending order, but put void label last
    void_class_id = 19
    regular_classes = [cls for cls in pixel_counts.keys() if cls != void_class_id]
    regular_classes_sorted = sorted(regular_classes, key=lambda x: pixel_counts[x], reverse=True)
    
    # Add void label at the end if it exists
    sorted_classes = regular_classes_sorted
    if void_class_id in pixel_counts:
        sorted_classes.append(void_class_id)
    
    sorted_counts = [pixel_counts[class_id] for class_id in sorted_classes]
    sorted_names = [class_names.get(class_id, f'Class {class_id}') for class_id in sorted_classes]
    
    # Create bar plot with seaborn
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(x=range(len(sorted_classes)), y=sorted_counts, palette="viridis")
    plt.title('Class distribution in PASTIS Training dataset', fontsize=16, fontweight='bold')
    plt.xticks(range(len(sorted_classes)), sorted_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, count in enumerate(sorted_counts):
        if count >= 1_000_000:
            label = f'{count/1_000_000:.1f}M'
        elif count >= 1_000:
            label = f'{count/1_000:.0f}k'
        else:
            label = str(count)
        ax.text(i, count, label, ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nPixel count summary:")
    for class_id in sorted_classes:
        class_name = class_names.get(class_id, f'Class {class_id}')
        print(f"Class {class_id} ({class_name}): {pixel_counts[class_id]:,} pixels")
    ######## PASTIS ##########################################################
    
    
    ######### Swiss Sentinel-2 ######################################################
    
    # # Swiss Sentinel-2 class names mapping
    # swiss_class_names = {
    #     0: 'Background',  # Assuming class 0 is background
    #     1: 'SummerBarley',
    #     2: 'WinterBarley',
    #     3: 'Oat',
    #     4: 'Wheat',
    #     5: 'Grain',
    #     6: 'Maize',
    #     7: 'EinkornWheat',
    #     8: 'SummerWheat',
    #     9: 'WinterWheat',
    #     10: 'Rye',
    #     11: 'Spelt',
    #     12: 'Sugar_beets',
    #     13: 'Beets',
    #     14: 'Potatoes',
    #     15: 'SummerRapeseed',
    #     16: 'WinterRapeseed',
    #     17: 'Soy',
    #     18: 'Sunflowers',
    #     19: 'Linen',
    #     20: 'Hemp',
    #     21: 'Field bean',
    #     22: 'Peas',
    #     23: 'Lupine',
    #     24: 'Pumpkin',
    #     25: 'Tobacco',
    #     26: 'Sorghum',
    #     27: 'Vegetables',
    #     28: 'Chicory',
    #     29: 'Buckwheat',
    #     30: 'Berries',
    #     31: 'nan',
    #     32: 'Biodiversity encouragement area',
    #     33: 'Fallow',
    #     34: 'MixedCrop',
    #     35: 'Mustard',
    #     36: 'Meadow',
    #     37: 'Pasture',
    #     38: 'Legumes',
    #     39: 'Vines',
    #     40: 'Apples',
    #     41: 'Pears',
    #     42: 'StoneFruit',
    #     43: 'Hops',
    #     44: 'TreeCrop',
    #     45: 'Chestnut',
    #     46: 'Special cultures',
    #     47: 'Hedge',
    #     48: 'Multiple',
    #     49: 'Forest',
    #     50: 'Non agriculture',
    #     51: 'Waters',
    #     52: 'Gardens'
    # }
    
    # # Swiss Sentinel-2 pixel counts data
    # swiss_pixel_counts = {
    #     0: 145355984.0, 1: 84139.0, 2: 2640192.0, 3: 277793.0, 4: 1629399.0,
    #     5: 88809.0, 6: 6163415.0, 7: 38234.0, 8: 105769.0, 9: 6931192.0,
    #     10: 180045.0, 11: 788241.0, 12: 1554675.0, 13: 34979.0, 14: 1066641.0,
    #     15: 9426.0, 16: 2468307.0, 17: 285505.0, 18: 517357.0, 19: 22084.0,
    #     20: 0.0, 21: 75621.0, 22: 255364.0, 23: 33493.0, 24: 12404.0,
    #     25: 39083.0, 26: 56842.0, 27: 1243236.0, 28: 38028.0, 29: 8219.0,
    #     30: 109067.0, 31: 1263095.0, 32: 77713.0, 33: 279765.0, 34: 82556.0,
    #     35: 3497.0, 36: 57482128.0, 37: 48493096.0, 38: 25757.0, 39: 1218790.0,
    #     40: 361513.0, 41: 74016.0, 42: 151695.0, 43: 1653.0, 44: 60288.0,
    #     45: 37517.0, 46: 13852.0, 47: 560673.0, 48: 5824556.0, 49: 2386853.0,
    #     50: 172455.0, 51: 17130.0, 52: 9819.0
    # }
    
    # # Remove classes with 0 pixels
    # swiss_pixel_counts = {k: v for k, v in swiss_pixel_counts.items() if v > 0}
    
    # # Group classes with less than 200k pixels into "Other"
    # threshold = 200000
    # major_classes = {k: v for k, v in swiss_pixel_counts.items() if v >= threshold}
    # minor_classes = {k: v for k, v in swiss_pixel_counts.items() if v < threshold}
    
    # # Calculate total pixels for "Other" category
    # other_total = sum(minor_classes.values())
    
    # # Sort major classes by pixel count in descending order
    # swiss_sorted_classes = sorted(major_classes.keys(), key=lambda x: major_classes[x], reverse=True)
    # swiss_sorted_counts = [int(major_classes[class_id]) for class_id in swiss_sorted_classes]
    # swiss_sorted_names = [swiss_class_names.get(class_id, f'Class {class_id}') for class_id in swiss_sorted_classes]
    
    # # Add "Other" category at the end if there are minor classes
    # if other_total > 0:
    #     swiss_sorted_classes.append('Other')
    #     swiss_sorted_counts.append(int(other_total))
    #     swiss_sorted_names.append('Other')
    
    # # Create bar plot for Swiss Sentinel-2 with seaborn
    # plt.figure(figsize=(16, 10))
    # ax = sns.barplot(x=range(len(swiss_sorted_classes)), y=swiss_sorted_counts, palette="viridis")
    # plt.title('Class distribution in Swiss Training dataset', fontsize=16, fontweight='bold')
    # plt.xticks(range(len(swiss_sorted_classes)), swiss_sorted_names, rotation=45, ha='right')
    
    # # Add value labels on bars
    # for i, count in enumerate(swiss_sorted_counts):
    #     if count >= 1_000_000:
    #         label = f'{count/1_000_000:.1f}M'
    #     elif count >= 1_000:
    #         label = f'{count/1_000:.0f}k'
    #     else:
    #         label = str(count)
    #     ax.text(i, count, label, ha='center', va='bottom', fontsize=8)
    
    # plt.tight_layout()
    # plt.show()
    
    # # Print summary
    # print(f"\nSwiss Sentinel-2 pixel count summary:")
    # for i, class_id in enumerate(swiss_sorted_classes):
    #     if class_id == 'Other':
    #         print(f"Other ({len(minor_classes)} classes): {other_total:,.0f} pixels")
    #     else:
    #         class_name = swiss_class_names.get(class_id, f'Class {class_id}')
    #         print(f"Class {class_id} ({class_name}): {major_classes[class_id]:,.0f} pixels")
    
    ######### Swiss Sentinel-2 ######################################################