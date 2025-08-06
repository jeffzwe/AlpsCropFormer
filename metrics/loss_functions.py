import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from utils.config_files_utils import get_params_values
from copy import deepcopy
from utils.torch_utils import DEVICE


def get_loss(config, device, reduction='mean', class_weights=None):
    model_config = config['MODEL']
    loss_config = config['SOLVER']

    # print(loss_config['loss_function'])

    if type(loss_config['loss_function']) in [list, tuple]:
        loss_fun = []
        loss_types = deepcopy(loss_config['loss_function'])
        config_ = deepcopy(config)
        for loss_fun_type in loss_types:
            config_['SOLVER']['loss_function'] = loss_fun_type
            loss_fun.append(get_loss(config_, device, reduction=reduction))
        return loss_fun

    # Background vs Non-Background Binary Cross-Entropy Loss ------------------------------
    if loss_config['loss_function'] == 'background_binary_cross_entropy':
        pos_weight = get_params_values(config['SOLVER'], 'pos_weight', None)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        return BackgroundBinaryCrossEntropy(reduction=reduction, pos_weight=pos_weight)

    # Cross-Entropy Loss ------------------------------------------------------------------
    elif loss_config['loss_function'] == 'cross_entropy':
        num_classes = get_params_values(model_config, 'num_classes', None)
        weight = torch.Tensor(num_classes * [1.0]).to(device)
        if class_weights is not None:
            weight = torch.Tensor(class_weights)
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction).to(device)
    
    # Masked Cross-Entropy Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_cross_entropy':
        mean = reduction == 'mean'
        return MaskedCrossEntropyLoss(mean=mean)
    
    # Focal Loss --------------------------------------------------------------------------
    elif loss_config['loss_function'] == ['focal_loss']:
        gamma = get_params_values(loss_config, "gamma", 1.0)
        num_classes = get_params_values(model_config, 'num_classes', None)
        weights = torch.Tensor(num_classes * [1.0]).to(device)
        if class_weights is not None:
            weights = torch.Tensor(class_weights)
        if loss_config['loss_function'] == 'focal_loss':
            return FocalLoss(gamma=gamma, alpha= weights, reduction=reduction, num_classes=model_config['num_classes'])

    # Masked Multiclass Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_dice_loss':
        return MaskedDiceLoss(reduction=reduction)


def per_class_loss(criterion, logits, labels, unk_masks, n_classes):
    class_loss = []
    class_counts = []
    for class_ in range(n_classes):
        idx = labels == class_
        class_loss.append(
            criterion(logits[idx.repeat(1, 1, 1, n_classes)].reshape(-1, n_classes),  # ???
                      labels[idx].reshape(-1, 1),
                      unk_masks[idx].reshape(-1, 1)).detach().cpu().numpy()
        )
        class_counts.append(unk_masks[idx].sum().cpu().numpy())
    class_loss = np.array(class_loss)
    class_counts = np.array(class_counts)
    return np.nan_to_num(class_loss, nan=0.0), class_counts


class BackgroundBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, reduction="mean", pos_weight=None):
        """
        Binary cross-entropy for background (0) vs non-background (!=0) classification.
        Automatically converts multi-class labels to binary format.
        """
        super(BackgroundBinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits, ground_truth):
        """
        Args:
            logits: (N, 1, H, W) - single channel output for binary classification
            ground_truth: (N, H, W) or tuple with mask - multi-class labels where 0=background
        """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        else:
            target = ground_truth[0]
            mask = ground_truth[1]
        
        # Convert multi-class labels to binary: 0 stays 0, everything else becomes 1
        binary_target = (target != 0).float()
        
        # Squeeze logits to match binary target shape
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # (N, H, W)
        
        if mask is not None:
            binary_target = binary_target[mask]
            logits = logits[mask]
            
        return self.loss_fn(logits, binary_target)


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean
    
    def forward(self, logits, ground_truth):
        """
            Args:
                logits: (N,T,H,W,...,NumClasses)A Variable containing a FloatTensor of size
                    (batch, max_len, num_classes) which contains the
                    unnormalized probability for each class.
                target: A Variable containing a LongTensor of size
                    (batch, max_len) which contains the index of the true
                    class for each corresponding step.
                length: A Variable containing a LongTensor of size (batch,)
                    which contains the length of each data in a batch.
            Returns:
                loss: An average loss value masked by the length.
            """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")
        
        logits = logits.permute(0, 2, 3, 1)
        # OG version
        if mask is not None:
            mask_flat = mask.reshape(-1, 1)  # (N*H*W x 1)
            nclasses = logits.shape[-1]
            logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
            target_flat = target.reshape(-1, 1)  # (N*H*W x 1)
            masked_target_flat = target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
        else:
            masked_logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_target_flat = target.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)
        masked_log_probs_flat = torch.nn.functional.log_softmax(masked_logits_flat, dim=1)  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat
        
        # # Use PyTorch's optimized cross_entropy with ignore_index for masking
        # if mask is not None:
        #     target_masked = target.clone()
        #     target_masked[~mask] = -100  # PyTorch's ignore_index
        #     loss = F.cross_entropy(
        #         logits.reshape(-1, logits.size(-1)), 
        #         target_masked.reshape(-1), 
        #         reduction='none'
        #     )
        #     loss = loss[target_masked.reshape(-1) != -100].unsqueeze(1)  # Match (M, 1) shape
        # else:
        #     loss = F.cross_entropy(
        #         logits.reshape(-1, logits.size(-1)), 
        #         target.reshape(-1), 
        #         reduction='none'
        #     )
        #     loss = loss.unsqueeze(1)  # Match original (N*H*W, 1)
        # if self.mean:
        #     return loss.mean()
        # return loss


class FocalLoss(nn.Module):
        
    def __init__(self, gamma=0, alpha=None, reduction=None, num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(self.alpha, list): self.alpha = torch.Tensor(self.alpha)
        
    
    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W)
        targets: (N, H, W)
        """
        N, C, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)     # (N*H*W, C)
        targets = targets.view(-1)                             # (N*H*W,)

        log_probs = F.log_softmax(inputs, dim=1)               # (N*H*W, C)
        probs = torch.exp(log_probs)

        log_p_t = log_probs[torch.arange(len(targets)), targets]  # (N*H*W,)
        p_t = probs[torch.arange(len(targets)), targets]          # (N*H*W,)

        focal_weight = (1 - p_t) ** self.gamma
        loss = -log_p_t * focal_weight

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            loss *= alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.view(N, H, W)  # shape: (N, H, W), if no reduction
    
    
class MaskedDiceLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, reduction=None):
        super(MaskedDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1, 1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        target_onehot = torch.eye(logits.shape[-1])[target].to(torch.float32).to(DEVICE)  # .permute(0,3,1,2).float().cuda()
        predicted_prob = F.softmax(logits, dim=-1)

        inter = (predicted_prob * target_onehot).sum(dim=-1)
        union = predicted_prob.pow(2).sum(dim=-1) + target_onehot.sum(dim=-1)

        loss = 1 - 2 * inter / union

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")
