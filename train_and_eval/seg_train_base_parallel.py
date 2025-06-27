import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_distributed_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
from tqdm import tqdm


def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
    """Single training step"""
    optimizer.zero_grad()
    outputs = net(sample['inputs'].to(device))
    outputs = outputs.permute(0, 2, 3, 1)
    ground_truth = loss_input_fn(sample, device)
    loss = loss_fn['mean'](outputs, ground_truth)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))
    optimizer.step()
    return outputs, ground_truth, loss, total_norm


def evaluate(net, evalloader, loss_fn, device, loss_input_fn, config, is_distributed=False):
    """Evaluation function with distributed support"""
    num_classes = config['MODEL']['num_classes']
    predicted_all = []
    labels_all = []
    losses_all = []
    
    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(evalloader):
            logits = net(sample['inputs'].to(device))
            logits = logits.permute(0, 2, 3, 1)
            _, predicted = torch.max(logits.data, -1)
            ground_truth = loss_input_fn(sample, device)
            loss = loss_fn['all'](logits, ground_truth)
            target, mask = ground_truth
            
            if mask is not None:
                predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
            else:
                predicted_all.append(predicted.view(-1).cpu().numpy())
                labels_all.append(target.view(-1).cpu().numpy())
            losses_all.append(loss.view(-1).cpu().detach().numpy())

    # Gather results from all processes if distributed
    if is_distributed:
        # Convert to tensors for all_gather
        predicted_tensor = torch.cat([torch.from_numpy(p) for p in predicted_all])
        labels_tensor = torch.cat([torch.from_numpy(l) for l in labels_all])
        losses_tensor = torch.cat([torch.from_numpy(l) for l in losses_all])
        
        # Gather from all processes
        gathered_predicted = [torch.zeros_like(predicted_tensor) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(dist.get_world_size())]
        gathered_losses = [torch.zeros_like(losses_tensor) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_predicted, predicted_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
        dist.all_gather(gathered_losses, losses_tensor)
        
        predicted_classes = torch.cat(gathered_predicted).numpy()
        target_classes = torch.cat(gathered_labels).numpy()
        losses = torch.cat(gathered_losses).numpy()
    else:
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

    eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                              n_classes=num_classes, unk_masks=None)

    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

    un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

    # Only print detailed results on main process
    if not is_distributed or dist.get_rank() == 0:
        print("-" * 145)
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print("-" * 145)

    return (un_labels,
            {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                       "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
             "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                       "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
             "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                       "Recall": class_recall, "F1": class_F1, "IOU": class_IOU}}
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
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1
    
    # For test mode, checkpoint is required
    if test_only and not checkpoint:
        raise ValueError("Checkpoint must be provided for test-only mode")
    
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    if is_main_process:
        print("Device: ", device)
        print("World size: ", world_size)
        if test_only:
            print("Running in TEST-ONLY mode")
        else:
            print("Current learn rate: ", lr)

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

    loss_input_fn = get_loss_data_input(config)
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

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
        
        if is_main_process:
            print("="*100)
            print("TEST RESULTS:")
            print("="*100)
            print("Test Loss: %.7f" % test_metrics[1]['macro']['Loss'])
            print("Test Macro - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, IOU: %.4f" % 
                  (test_metrics[1]['macro']['Accuracy'], test_metrics[1]['macro']['Precision'],
                   test_metrics[1]['macro']['Recall'], test_metrics[1]['macro']['F1'], test_metrics[1]['macro']['IOU']))
            print("Test Micro - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, IOU: %.4f" % 
                  (test_metrics[1]['micro']['Accuracy'], test_metrics[1]['micro']['Precision'],
                   test_metrics[1]['micro']['Recall'], test_metrics[1]['micro']['F1'], test_metrics[1]['micro']['IOU']))
            print("="*100)
            
            # Save test results to file
            if save_path:
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
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = None
    if is_main_process:
        writer = SummaryWriter(save_path)

    BEST_IOU = 0

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        
        # Set epoch for distributed sampler
        if is_distributed:
            dataloaders['train'].sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\nEpoch {epoch}/{num_epochs}")
            pbar = tqdm(total=num_steps_train, desc=f"Training Epoch {epoch}")
        
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            logits, ground_truth, loss, grad_norm = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn)
            
            # Logging only on main process
            if is_main_process and writer:
                writer.add_scalar('training_gradient_norm', grad_norm, abs_step)
            
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None

            # Print batch statistics on main process
            if abs_step % train_metrics_steps == 0 and is_main_process:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, 
                    loss=loss, epoch=epoch, step=step)
                if writer:
                    write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy']))

            # Save checkpoints only on main process
            if abs_step % save_steps == 0 and is_main_process:
                model_to_save = net.module if is_distributed else net
                torch.save(model_to_save.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))

            # Evaluate model
            if abs_step % eval_steps == 0:
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, device, loss_input_fn, config, is_distributed)
                
                # Synchronize before handling results
                if is_distributed:
                    dist.barrier()
                
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
                    
                    print("Evaluation - Loss: %.7f, Macro IOU: %.4f, Micro IOU: %.4f" % 
                          (eval_metrics[1]['macro']['Loss'], eval_metrics[1]['macro']['IOU'], eval_metrics[1]['micro']['IOU']))

                net.train()
            
            if is_main_process:
                pbar.update(1)

        scheduler.step_update(abs_step)
        
        if is_main_process:
            pbar.close()


def main_worker(rank, world_size, config, args):
    """Worker function for each process"""
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = get_device(device_ids, allow_cpu=True)
    
    try:
        dataloaders = get_distributed_dataloaders(config, world_size, rank)
        net = get_model(config, device)
        train_and_evaluate(net, dataloaders, config, device, rank, world_size, args.test)
    except Exception as e:
        print(f"Error in worker {rank}: {e}")
        raise
    finally:
        if world_size > 1:
            cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config_file', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str, help='gpu ids to use')
    parser.add_argument('--test', action='store_true', help='run test evaluation only (requires checkpoint)')

    args = parser.parse_args()
    
    # Validate arguments
    if args.test and not args.config_file:
        parser.error("--test requires --config_file")
    
    config_file = args.config_file
    device_ids = [int(d) for d in args.device.split(',')]
    
    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    
    world_size = len(device_ids)
    
    if world_size > 1:
        # Multi-GPU distributed training/testing
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training/testing
        main_worker(0, 1, config, args)
