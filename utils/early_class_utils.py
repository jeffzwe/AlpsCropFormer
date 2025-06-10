import torch
import random

def early_class_trunc(sample):
    """
    Randomly truncates each input sequence in the batch for early classification.
    
    Args:
        sample (dict): Dictionary containing 'inputs' and 'org_len' keys
                       'inputs': Tensor of shape [Batch, Timesteps, H, W, C]
                       'org_len': Original lengths of each sequence before padding
    
    Returns:
        dict: Modified sample with truncated inputs and weight ratios
    """
    # Modify inputs directly without creating a copy
    inputs = sample['inputs']
    org_lengths = sample['seq_lengths']
    batch_size = inputs.shape[0]
    
    # Create a tensor to store weight ratios
    weight_ratios = torch.zeros(batch_size, dtype=torch.float32)
    
    # For each item in the batch
    for i in range(batch_size):
        org_len = org_lengths[i].item()
        # Select a random truncation point between 3 and the original length
        trunc_point = random.randint(3, org_len)
        
        # Zero out timesteps beyond the truncation point
        inputs[i, trunc_point:, :, :, :] = 0.0
        
        # Calculate weight ratio:
        weight_ratios[i] = (org_len - trunc_point + 1) / org_len
    
    # Add weight ratios to the sample dictionary
    sample['weight_ratio'] = weight_ratios
    
    return sample
