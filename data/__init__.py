import os
from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
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
    train_config['paths'] = os.path.join(train_config['base_dir'], DATASET_INFO[train_config['dataset']]['paths_train'])
    if 'PASTIS' in train_config['dataset']:
        dataloaders['train'] = get_pastis_dataloader(
            paths_file=train_config['paths'], root_dir=train_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=True),
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])
        
    # EVAL data --------------------------------------------------------------------------------------------------------
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    eval_config['paths'] = os.path.join(eval_config['base_dir'], DATASET_INFO[eval_config['dataset']]['paths_eval'])
    if 'PASTIS' in eval_config['dataset']:
        dataloaders['eval'] = get_pastis_dataloader(
            paths_file=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'])
        
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
