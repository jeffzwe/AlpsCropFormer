import os
from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.dataloader import get_distr_dataloader as get_pastis_distr_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from data.Sentinel.dataloader_webdataset import get_dataloader as get_sentinel_dataloader
from data.Sentinel.dataloader_webdataset import get_distr_dataloader as get_sentinel_distr_dataloader
from utils.config_files_utils import read_yaml


DATASET_INFO = read_yaml("data/datasets.yaml")

def get_dataloaders(config):

    model_config = config['MODEL']
    train_config = config['DATASETS']['train']
    eval_config  = config['DATASETS']['eval']
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
        train_config['webdataset_dir'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['paths_train'])
        
        dataloaders['train'] = get_sentinel_dataloader(
            webdataset_dir=train_config['webdataset_dir'],
            temp_length=model_config['max_seq_len'],
            truncate_month=model_config['truncate_month'],
            timestamp_mode=model_config['timestamp_mode'],
            is_training=True,
            batch_size=train_config['batch_size'],
            num_workers=train_config['num_workers'],
            shuffle=True
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
        eval_config['webdataset_dir'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['paths_eval'])
        
        dataloaders['eval'] = get_sentinel_dataloader(
            webdataset_dir=eval_config['webdataset_dir'],
            temp_length=model_config['max_seq_len'],
            truncate_month=model_config['truncate_month'],
            timestamp_mode=model_config['timestamp_mode'],
            is_training=False,
            batch_size=eval_config['batch_size'],
            num_workers=eval_config['num_workers'],
            shuffle=False
        )
        
    # TEST data --------------------------------------------------------------------------------------------------------
    if 'test' in config['DATASETS']:
        test_config = config['DATASETS']['test']
        test_config['base_dir'] = DATASET_INFO[test_config['dataset']]['basedir']
        
        if 'PASTIS' in test_config['dataset']:
            test_config['paths'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['paths_test'])
            dataloaders['test'] = get_pastis_dataloader(
                paths_file=test_config['paths'], root_dir=test_config['base_dir'],
                transform=PASTIS_segmentation_transform(model_config, is_training=False),
                batch_size=test_config['batch_size'], shuffle=False, num_workers=test_config['num_workers'])
        elif 'Sentinel' in test_config['dataset']:
            test_config['webdataset_dir'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['paths_test'])
            
            dataloaders['test'] = get_sentinel_dataloader(
                webdataset_dir=test_config['webdataset_dir'],
                temp_length=model_config['max_seq_len'],
                truncate_month=model_config['truncate_month'],
                timestamp_mode=model_config['timestamp_mode'],
                is_training=False,
                batch_size=test_config['batch_size'],
                num_workers=test_config['num_workers'],
                shuffle=False
            )
    
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
        train_config['webdataset_dir'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['paths_train'])
        
        dataloaders['train'] = get_sentinel_distr_dataloader(
            webdataset_dir=train_config['webdataset_dir'],
            temp_length=model_config['max_seq_len'],
            truncate_month=model_config['truncate_month'],
            timestamp_mode=model_config['timestamp_mode'],
            is_training=True,
            world_size=world_size,
            rank=rank,
            batch_size=train_config['batch_size'],
            num_workers=train_config['num_workers']
        )
    
    # EVAL data with distributed sampler
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    if 'Sentinel' in eval_config['dataset']:
        eval_config['webdataset_dir'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['paths_eval'])
        
        dataloaders['eval'] = get_sentinel_distr_dataloader(
            webdataset_dir=eval_config['webdataset_dir'],
            temp_length=model_config['max_seq_len'],
            truncate_month=model_config['truncate_month'],
            timestamp_mode=model_config['timestamp_mode'],
            is_training=False,
            world_size=world_size,
            rank=rank,
            batch_size=eval_config['batch_size'],
            num_workers=eval_config['num_workers']
        )
    
    # TEST data with distributed sampler
    if 'test' in config['DATASETS']:
        test_config = config['DATASETS']['test']
        test_config['base_dir'] = DATASET_INFO[test_config['dataset']]['basedir']
        
        if 'PASTIS' in test_config['dataset']:
            test_config['paths'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['paths_test'])
            dataloaders['test'] = get_pastis_distr_dataloader(
                paths_file=test_config['paths'], root_dir=test_config['base_dir'],
                transform=PASTIS_segmentation_transform(model_config, is_training=False),
                world_size=world_size, rank=rank,
                batch_size=test_config['batch_size'], num_workers=test_config['num_workers'])
        elif 'Sentinel' in test_config['dataset']:
            test_config['webdataset_dir'] = os.path.join(test_config['base_dir'], DATASET_INFO[test_config['dataset']]['paths_test'])
            
            dataloaders['test'] = get_sentinel_distr_dataloader(
                webdataset_dir=test_config['webdataset_dir'],
                temp_length=model_config['max_seq_len'],
                truncate_month=model_config['truncate_month'],
                timestamp_mode=model_config['timestamp_mode'],
                is_training=False,
                world_size=world_size,
                rank=rank,
                batch_size=test_config['batch_size'],
                num_workers=test_config['num_workers']
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
