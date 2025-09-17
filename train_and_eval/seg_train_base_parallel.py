import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_net_trainable_params, load_from_checkpoint
from data import get_distributed_dataloaders
from metrics.torch_metrics import get_mean_metrics, get_binary_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss, get_top_predictions_per_class
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
from data.Sentinel.dataloader_webdataset import get_class_weights, get_aggregated_class_weights

def setup_output_redirection(rank, save_path):
    """Redirect stdout to rank-specific output file"""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, f"rank_{rank}_output.log")
        sys.stdout = open(output_file, 'w', buffering=1)  # Line buffered
        print(f"Output redirected to {output_file}")
    else:
        print(f"Warning: No save_path provided, output not redirected for rank {rank}")

def cleanup_output_redirection():
    """Close output file and restore stdout"""
    if hasattr(sys.stdout, 'close') and sys.stdout != sys.__stdout__:
        sys.stdout.close()
    sys.stdout = sys.__stdout__

def setup_ddp(rank, world_size):
    master_addr = os.environ.get('MASTER_ADDR')
    master_port = os.environ.get('MASTER_PORT')
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn, epoch):
    """Single training step"""
    optimizer.zero_grad()
    outputs = net(sample['inputs'].to(device))
    ground_truth = loss_input_fn(sample, device)
    loss = loss_fn['mean'](outputs, ground_truth)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))
    optimizer.step()
    return outputs, ground_truth, loss, total_norm


def evaluate(net, evalloader, loss_fn, device, loss_input_fn, config, is_distributed=False):
    """Evaluation function with distributed support"""
    num_classes = config['MODEL']['num_classes']
    loss_function = config['SOLVER']['loss_function']
    predicted_all = []
    labels_all = []
    losses_all = []
    
    rank = dist.get_rank() if is_distributed else -1
    
    net.eval()
    
    with torch.no_grad():
        for sample in evalloader:
            logits = net(sample['inputs'].to(device))
            _, predicted = torch.max(logits.data, 1)
            ground_truth = loss_input_fn(sample, device)
            loss = loss_fn['all'](logits, ground_truth)
            
            if loss_function in ['masked_cross_entropy', 'masked_focal_loss']:
                target, mask = ground_truth
                predicted_all.append(predicted.flatten()[mask.flatten()].cpu().numpy())
                labels_all.append(target.flatten()[mask.flatten()].cpu().numpy())
            else:
                predicted_all.append(predicted.detach().cpu().flatten().numpy())
                labels_all.append(ground_truth.detach().cpu().flatten().numpy())
            losses_all.append(loss.detach().cpu().flatten().numpy())
            
    print(f"[Rank {rank}] Evaluation complete")

    # Gather results from all processes if distributed
    if is_distributed:
        local_data = {
            'preds': np.concatenate(predicted_all),
            'labels': np.concatenate(labels_all),
            'losses': np.concatenate(losses_all)
        }
        gathered_data = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_data, local_data)

        if dist.get_rank() == 0:
            predicted_classes = np.concatenate([d['preds'] for d in gathered_data])
            target_classes = np.concatenate([d['labels'] for d in gathered_data])
            losses = np.concatenate([d['losses'] for d in gathered_data])
        else:
            return
    else:
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)
        
    eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                              n_classes=num_classes)
    
    
    top_preds = get_top_predictions_per_class(predicted_classes, target_classes, num_classes, top_x=5)
 
    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

    un_labels, class_loss = get_per_class_loss(losses, target_classes)

    # Remove detailed print statement - only log essential info
    if not is_distributed or dist.get_rank() == 0:
        print(f"Eval complete - Loss: {losses.mean():.6f}, Micro IOU: {micro_IOU:.4f}, Micro Acc: {micro_acc:.4f}")

    return (np.arange(num_classes),
            {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                       "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
             "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                       "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
             "class": {"Accuracy": class_acc, "Precision": class_precision,
                       "Recall": class_recall, "F1": class_F1, "IOU": class_IOU}},
            top_preds
            )

def train_and_evaluate(net, dataloaders, config, device, rank=0, world_size=1, test_only=False):
    """Main training and evaluation function with DDP support"""
    
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    # Configuration
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    start_epoch = config['CHECKPOINT'].get("start_epoch", 1)
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    class_weights_p = config['SOLVER'].get('class_weights', None)
    
    # For test mode, checkpoint is required
    if test_only and not checkpoint:
        raise ValueError("Checkpoint must be provided for test-only mode")
    
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    if is_main_process:
        print("Device: ", device)
        print("World size: ", world_size)
        print(f"Batch size: {config['DATASETS']['train']['batch_size']}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Learning rate: {lr}")
        print(f"Dropout rate: {config['MODEL']['dropout']}")
        print(f"Embedding Dropout rate: {config['MODEL']['emb_dropout']}")
        print(f"Weight decay: {weight_decay}")
        print(f"Temporal length: {config['MODEL']['max_seq_len']}")
        print(f"Loss function: {config['SOLVER']['loss_function']}")
        print(f"timestamp_mode: {config['MODEL']['timestamp_mode']}")
        print(f"truncate_month: {config['MODEL']['truncate_month']}")
        print(f"Number of workers: {config['DATASETS']['train']['num_workers']}")
        if test_only:
            print("Running in TEST-ONLY mode")
        else:
            print("Training at current learn rate: ", lr)

    # Move model to device before DDP wrapping
    net.to(device)
    
    # Wrap model with DDP
    if is_distributed:
        net = DDP(net, device_ids=[rank])

    # Create save directory only on main process
    if save_path and is_main_process and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    if is_main_process and not test_only:
        copy_yaml(config)

    class_weights = get_class_weights('lnf_code_2022_pixel_counts.txt', 'crop_mappings.csv', power=class_weights_p)
    # class_weights = get_aggregated_class_weights(0.1)

    loss_input_fn = get_loss_data_input(config)
    loss_fn = {'all': get_loss(config, device, reduction='none', class_weights=class_weights),
               'mean': get_loss(config, device, reduction="mean", class_weights=class_weights)}

    # Test-only mode: run evaluation on test set and exit
    if test_only:
        if 'test' not in dataloaders:
            raise ValueError("Test dataloader not found. Make sure test data is configured.")
        
        if is_main_process:
            print("Running test evaluation...")
        
        test_metrics = evaluate(net, dataloaders['test'], loss_fn, device, loss_input_fn, config, is_distributed)
        
        # Synchronize before printing results
        if is_distributed:
            dist.barrier()
        
        # Save test results to file
        if is_main_process and save_path:
            results_file = os.path.join(save_path, "test_results.txt")
            with open(results_file, 'w') as f:
                f.write("TEST RESULTS:\n")
                f.write("="*50 + "\n")
                f.write("Test Loss: %.7f\n" % test_metrics[1]['macro']['Loss'])
                f.write("Test Macro - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, IOU: %.4f\n" % 
                        (test_metrics[1]['macro']['Accuracy'], test_metrics[1]['macro']['Precision'],
                            test_metrics[1]['macro']['Recall'], test_metrics[1]['macro']['F1'], test_metrics[1]['macro']['IOU']))
                f.write("Test Micro - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, IOU: %.4f\n" % 
                        (test_metrics[1]['micro']['Accuracy'], test_metrics[1]['micro']['Precision'],
                            test_metrics[1]['micro']['Recall'], test_metrics[1]['micro']['F1'], test_metrics[1]['micro']['IOU']))
            print(f"Test results saved to: {results_file}")
    
        return test_metrics

    num_steps_train = len(dataloaders['train'])
    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, eps=1e-8)
    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = None
    if is_main_process:
        writer = SummaryWriter(save_path)

    BEST_IOU = 0

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        
        # Set epoch for distributed sampler. DONT USE WITH WEBDATASET
        # if is_distributed:
        #     dataloaders['train'].sampler.set_epoch(epoch)
        abs_step = (epoch-1) * num_steps_train
        
        print(f"\nStarting Epoch {epoch}/{start_epoch + num_epochs-1}")
        epoch_start_time = time.time()
        progress_log_interval = max(1, num_steps_train // 4)  # Log progress every 10% of steps
        
        for step, sample in enumerate(dataloaders['train']):
            abs_step += 1
            logits, ground_truth, loss, grad_norm = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn, epoch)
            
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None

            # Progress logging - less frequent than before
            if step % progress_log_interval == 0:
                elapsed = time.time() - epoch_start_time
                elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
                progress = (step + 1) / num_steps_train * 100
                eta = elapsed / (step + 1) * (num_steps_train - step - 1)
                print(f"  Progress: {progress:5.1f}% | Step {step+1}/{num_steps_train} | "
                      f"Elapsed: {elapsed_str} | ETA: {eta/60:.1f}m")

            # Batch metrics logging - only to tensorboard, no print
            if is_main_process and abs_step % train_metrics_steps == 0:
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, 
                    loss=loss, epoch=epoch, step=step)
                if is_main_process and writer:
                    write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                    writer.add_scalar('training_gradient_norm', grad_norm, abs_step)
                    print(f"    Batch metrics logged at step {abs_step}")

            # Save checkpoints only on main process
            if is_main_process and abs_step % save_steps == 0:
                model_to_save = net.module if is_distributed else net
                torch.save(model_to_save.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                print(f"  Checkpoint saved at step {abs_step}")

            # Evaluate model
            if abs_step % eval_steps == 0:
                print(f"  Running evaluation at step {abs_step}...")
                
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, device, loss_input_fn, config, is_distributed)
                
                # Only main process handles evaluation logging and best model saving
                if is_main_process:
                    current_iou = eval_metrics[1]['macro']['IOU']
                    if current_iou > BEST_IOU:
                        model_to_save = net.module if is_distributed else net
                        torch.save(model_to_save.state_dict(), "%s/best.pth" % (save_path))
                        BEST_IOU = current_iou

                    if writer:
                        write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                        write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                        write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval", optimizer=None)
                        
                    if epoch % 5 == 0:
                        top_preds = eval_metrics[2]
                        for true_class, data in top_preds.items():
                            print(f"True class {true_class} (total samples: {data['total_samples']}):")
                            for i, (pred_class, count, percentage) in enumerate(zip(data['top_classes'], data['counts'], data['percentages'])):
                                status = "(correct)" if pred_class == true_class else "(misclassified)"
                                print(f"  Top {i+1}: predicted as class {pred_class} ({count} times, {percentage:.1f}%) {status}")

                torch.cuda.empty_cache()
                net.train()
        
        # End of epoch logging
        epoch_elapsed = time.time() - epoch_start_time
        epoch_elapsed_str = f"{int(epoch_elapsed//3600):02d}:{int((epoch_elapsed%3600)//60):02d}:{int(epoch_elapsed%60):02d}"
        print(f"Epoch {epoch} completed in {epoch_elapsed_str} | Best IOU: {BEST_IOU:.4f}")

        scheduler.step_update(abs_step)


def main_worker(rank, world_size, config, args):
    """Worker function for each process"""
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    save_path = f'output_logs/slurm-{os.environ.get("SLURM_JOB_ID")}'
    setup_output_redirection(rank, save_path)
    
    try:
        dataloaders = get_distributed_dataloaders(config, world_size, rank)
        net = get_model(config, device)
        train_and_evaluate(net, dataloaders, config, device, rank, world_size, args.test)
        
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        raise
    finally:
        cleanup_output_redirection()
        if world_size > 1:
            cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config_file', help='configuration (.yaml) file to use')
    parser.add_argument('--test', action='store_true', help='run test evaluation only (requires checkpoint)')

    args = parser.parse_args()
    
    # Validate arguments
    if args.test and not args.config_file:
        parser.error("--test requires --config_file")
    
    config_file = args.config_file
    
    config = read_yaml(config_file)
    
    # Read env variables from SLURM
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    main_worker(local_rank, world_size, config, args)
