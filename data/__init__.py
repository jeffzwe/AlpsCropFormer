import os
from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.dataloader import get_distr_dataloader as get_pastis_distr_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from data.Sentinel.dataloader import get_dataloader as get_sentinel_dataloader
from data.Sentinel.dataloader import get_distr_dataloader as get_sentinel_distr_dataloader
from utils.config_files_utils import read_yaml


DATASET_INFO = read_yaml("data/datasets.yaml")

def get_dataloaders(config):

    model_config = config['MODEL']
    train_config = config['DATASETS']['train']
    train_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config  = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    dataloaders = {}
    
    # TRAIN data -------------------------------------------------------------------------------------------------------
    train_config['base_dir'] = DATASET_INFO[train_config['dataset']]['basedir']
    if 'PASTIS' in train_config['dataset']:
        train_config['paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['paths_train'])
        dataloaders['train'] = get_pastis_dataloader(
            paths_file=train_config['paths'], root_dir=train_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=True),
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])
    elif 'Sentinel' in train_config['dataset']:
        train_config['crop_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['crop_train'])
        train_config['gt_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['gt_train'])
        train_config['temp_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['temp_train'])
        train_config['crop_map'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['crop_map'])
        
        dataloaders['train'] = get_sentinel_dataloader(
            crop_path=train_config['crop_paths'], gt_path=train_config['gt_paths'],
            temp_path=train_config['temp_paths'], crop_map=train_config['crop_map'], truncate_portion= model_config['truncate_portion'],
            timestamp_mode=model_config['timestamp_mode'], temp_length= model_config['max_seq_len'], cropping_mode=model_config['cropping_mode'],
            img_res = model_config['img_res'], batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'],
            is_training=True
        )
        
    # EVAL data --------------------------------------------------------------------------------------------------------
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    if 'PASTIS' in eval_config['dataset']:
        eval_config['paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['paths_eval'])
        dataloaders['eval'] = get_pastis_dataloader(
            paths_file=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'])
    elif 'Sentinel' in eval_config['dataset']:
        eval_config['crop_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['crop_train'])
        eval_config['gt_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['gt_train'])
        eval_config['temp_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['temp_train'])
        eval_config['crop_map'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['crop_map'])
        
        dataloaders['eval'] = get_sentinel_dataloader(
            crop_path=eval_config['crop_paths'], gt_path=eval_config['gt_paths'],
            temp_path=eval_config['temp_paths'], crop_map=eval_config['crop_map'], truncate_portion= model_config['truncate_portion'],
            timestamp_mode=model_config['timestamp_mode'], temp_length= model_config['max_seq_len'], cropping_mode=model_config['cropping_mode'],
            img_res = model_config['img_res'], batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'],
            is_training=False
        )
        
    # TEST data --------------------------------------------------------------------------------------------------------
    
    return dataloaders

def get_distributed_dataloaders(config, world_size, rank):
    """Get dataloaders with distributed samplers"""
    model_config = config['MODEL']
    train_config = config['DATASETS']['train']
    eval_config = config['DATASETS']['eval']
    
    dataloaders = {}
    
    # TRAIN data with distributed sampler
    train_config['base_dir'] = DATASET_INFO[train_config['dataset']]['basedir']
    if 'Sentinel' in train_config['dataset']:
        train_config['crop_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['crop_train'])
        train_config['gt_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['gt_train'])
        train_config['temp_paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['temp_train'])
        train_config['crop_map'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['crop_map'])
        
        dataloaders['train'] = get_sentinel_distr_dataloader(
            crop_path=train_config['crop_paths'], gt_path=train_config['gt_paths'],
            temp_path=train_config['temp_paths'], crop_map=train_config['crop_map'],
            truncate_portion=model_config['truncate_portion'],
            timestamp_mode=model_config['timestamp_mode'], temp_length=model_config['max_seq_len'],
            cropping_mode=model_config['cropping_mode'], img_res=model_config['img_res'],
            is_training=True, world_size=world_size, rank=rank,
            batch_size=train_config['batch_size'], num_workers=train_config['num_workers']
        )
    
    # EVAL data with distributed sampler
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    if 'Sentinel' in eval_config['dataset']:
        eval_config['crop_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['crop_train'])
        eval_config['gt_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['gt_train'])
        eval_config['temp_paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['temp_train'])
        eval_config['crop_map'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['crop_map'])
        
        dataloaders['eval'] = get_sentinel_distr_dataloader(
            crop_path=eval_config['crop_paths'], gt_path=eval_config['gt_paths'],
            temp_path=eval_config['temp_paths'], crop_map=eval_config['crop_map'],
            truncate_portion=model_config['truncate_portion'],
            timestamp_mode=model_config['timestamp_mode'], temp_length=model_config['max_seq_len'],
            cropping_mode=model_config['cropping_mode'], img_res=model_config['img_res'],
            is_training=False, world_size=world_size, rank=rank,
            batch_size=eval_config['batch_size'], num_workers=eval_config['num_workers']
        )
    
    return dataloaders


def get_loss_data_input(config):

    def segmentation_ground_truths(sample, device):
        labels = sample['labels'].to(device)
        if 'unk_masks' in sample.keys():
            unk_masks = sample['unk_masks'].to(device)
        else:
            unk_masks = None

        if 'edge_labels' in sample.keys():
            edge_labels = sample['edge_labels'].to(device)
            return labels, edge_labels, unk_masks
        return labels, unk_masks

    return segmentation_ground_truths
