import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from utils.early_class_utils import early_class_trunc
from data import get_loss_data_input
from tqdm import tqdm


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        sample = early_class_trunc(sample)
        outputs = net(sample['inputs'].to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn(outputs, ground_truth)
        
        # Scale loss by truncation_ratio if it exists in the sample
        truncation_ratio = sample['weight_ratio'].to(device)
        loss = loss * truncation_ratio
        loss = loss.mean()    
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))
        optimizer.step()
        return outputs, ground_truth, loss, total_norm

  
    def evaluate(net, evalloader, loss_fn, config, device):
        num_classes = config['MODEL']['num_classes']
        img_res = config['MODEL']['img_res']
        predicted_all = []
        labels_all = []
        losses_all = []
        
        # Parameters for dynamic inference
        min_timesteps = get_params_values(config['SOLVER'], "min_timesteps", 3)
        confidence_threshold = get_params_values(config['SOLVER'], "confidence_threshold", 0.9)
        
        # Track timestep usage ratios per sample
        timestep_ratios = []
        total_used_timesteps = []
        
        net.eval()
        with torch.no_grad():
            # Add tqdm progress bar for evaluation loop
            total_eval_steps = len(evalloader)
            with tqdm(total=total_eval_steps, desc="Evaluating") as eval_pbar:
                for step, sample in enumerate(evalloader):
                    batch_size = sample['inputs'].size(0)
                    
                    # Get the sequence length for each sample in the batch
                    seq_lengths = sample['seq_lengths']  # Keep on CPU, no need to move to device
                    
                    # Determine maximum sequence length in this batch
                    max_steps = seq_lengths.max().item()
                    
                    # Initialize tracking variables
                    confident_samples = torch.zeros(batch_size, dtype=torch.bool)
                    
                    # Track actual timesteps used for each sample in this batch
                    batch_used_timesteps = torch.zeros(batch_size, dtype=torch.int)
                    
                    # Storage for final predictions and metrics
                    batch_ground_truth = loss_input_fn(sample, device)
                    target, mask = batch_ground_truth  # Always a tuple with (target, mask)
                    
                    # Initialize predictions storage with correct shape [BatchSize, H, W]
                    batch_logits = torch.zeros((batch_size, img_res, img_res, num_classes), dtype=torch.float32, device=device)
                    batch_predictions = torch.zeros((batch_size, img_res, img_res), dtype=torch.long, device=device)
                    
                    # Move inputs to device
                    inputs = sample['inputs'].to(device)
                    
                    # Dynamic inference loop starting from minimum timesteps
                    curr_step = min_timesteps
                    
                    while (not confident_samples.all()) and curr_step <= max_steps:
                        # For non-confident samples that haven't reached their max sequence length
                        active_indices = []
                        active_inputs = []
                        
                        for i in range(batch_size):
                            if not confident_samples[i] and curr_step <= seq_lengths[i]:
                                active_indices.append(i)
                                
                                # Create truncated input by zeroing out timesteps beyond curr_step
                                # inputs[i:i+1] keeps the batch dimension, resulting in a 5D tensor [1, T, C, H, W]
                                truncated_input = inputs[i:i+1].clone()
                                truncated_input[:, curr_step:, :, :, :] = 0.0
                                active_inputs.append(truncated_input)
                        
                        if len(active_inputs) > 0:
                            # Combine all active samples into a mini-batch
                            current_inputs = torch.cat(active_inputs, dim=0)
                            
                            # Forward pass
                            logits = net(current_inputs)
                            logits = logits.permute(0, 2, 3, 1)
                            
                            # Get predictions and confidence
                            probs = torch.softmax(logits, dim=-1)
                            confidence, predicted = torch.max(probs, -1)
                            
                            # Get average confidence per sample
                            avg_confidence = confidence.mean(dim=[1, 2])
                            
                            # Process results for each active sample
                            for j, i in enumerate(active_indices):
                                # Store predictions for all active samples at current step
                                batch_logits[i] = logits[j:j+1]
                                batch_predictions[i] = predicted[j:j+1]
                                
                                # Check if confidence threshold is met or max steps reached
                                if avg_confidence[j] >= confidence_threshold or curr_step >= seq_lengths[i]:
                                    confident_samples[i] = True
                                    batch_used_timesteps[i] = curr_step  # Record the timestep where confidence was reached
                        
                        # Increment step counter for next iteration
                        curr_step += 1
                    
                    # For any remaining non-confident samples, use their maximum sequence length
                    for i in range(batch_size):
                        if not confident_samples[i]:
                            batch_used_timesteps[i] = seq_lengths[i].item()
                    
                    # Calculate and store timestep ratio for each sample in the batch
                    for i in range(batch_size):
                        timestep_ratio = batch_used_timesteps[i].item() / seq_lengths[i].item()
                        timestep_ratios.append(timestep_ratio)
                        total_used_timesteps.append(batch_used_timesteps[i].item())
                    
                    # Calculate loss only once for the entire batch with final predictions
                    loss = loss_fn(batch_logits, batch_ground_truth)
                    
                    # Process the batch results
                    if mask is not None:
                        predicted_all.append(batch_predictions.view(-1)[mask.view(-1)].cpu().numpy())
                        labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                    else:
                        predicted_all.append(batch_predictions.view(-1).cpu().numpy())
                        labels_all.append(target.view(-1).cpu().numpy())
                    losses_all.append(loss.view(-1).cpu().detach().numpy())
                    
                    # Update progress bar at the end of each batch
                    eval_pbar.update(1)

        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        # Calculate average timestep usage ratio across all samples
        avg_timestep_ratio = np.mean(timestep_ratios)
        avg_used_timesteps = np.mean(total_used_timesteps)
        
        print(f"Average timestep usage ratio: {avg_timestep_ratio:.4f}")
        
        
        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s, timestep usage avg_ratio: %.4f, timestep usage avg used: %.4f" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes), avg_timestep_ratio, avg_used_timesteps))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU},
                 "timestep_usage": {"Avg_Ratio": avg_timestep_ratio, "Used_Timesteps_Mean": avg_used_timesteps}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    
    # Check if early classification is enabled
    early_classification = get_params_values(config['SOLVER'], "early_classification", False)
    print("Early classification enabled:", early_classification)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
        
    print("Device: ", device)

    print("current learn rate: ", lr)
    
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)
    
    loss_fn = get_loss(config, device, reduction=None)

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = SummaryWriter(save_path)

    BEST_IOU = 0

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        with tqdm(total=num_steps_train, desc=f"Training Epoch {epoch}") as pbar:
            
            for step, sample in enumerate(dataloaders['train']):
                abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
                logits, ground_truth, loss, grad_norm = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn=loss_input_fn)
                
                # Log individual gradient norm to TensorBoard
                writer.add_scalar('training_gradient_norm', grad_norm, abs_step)
                
                if len(ground_truth) == 2:
                    labels, unk_masks = ground_truth
                else:
                    labels = ground_truth
                    unk_masks = None
                # print batch statistics ----------------------------------------------------------------------------------#
                if abs_step % train_metrics_steps == 0:
                    logits = logits.permute(0, 3, 1, 2)
                    batch_metrics = get_mean_metrics(
                        logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                        step=step)
                    write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                    print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                        "batch recall: %.4f, batch F1: %.4f" %
                        (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                        batch_metrics['Recall'], batch_metrics['F1']))

                if abs_step % save_steps == 0:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                    else:
                        torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))

                # evaluate model ------------------------------------------------------------------------------------------#
                if abs_step % eval_steps == 0:
                    eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config, device)
                    if eval_metrics[1]['macro']['IOU'] > BEST_IOU:
                        if len(local_device_ids) > 1:
                            torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                        else:
                            torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                        BEST_IOU = eval_metrics[1]['macro']['IOU']

                    # Write timestep usage metrics to tensorboard
                    writer.add_scalar('eval/timestep_ratio', eval_metrics[1]['timestep_usage']['Avg_Ratio'], abs_step)
                    writer.add_scalar('eval/used_timesteps_mean', eval_metrics[1]['timestep_usage']['Used_Timesteps_Mean'], abs_step)

                    write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                    write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                    write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval",
                                        optimizer=None)
                    net.train()
                    exit(0)
                pbar.update(1)

            scheduler.step_update(abs_step)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config_file', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                         help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config_file
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=True)  # Allow CPU for apple silicon compatibility

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)
    
    # train_iter = iter(dataloaders['train'])
    # sample = next(train_iter)
    
    # print("Mask Dim: ", sample['unk_masks'].shape)
    # print("Label Dim: ", sample['labels'].shape)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)
