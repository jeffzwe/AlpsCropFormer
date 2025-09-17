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

    # Binary Cross-Entropy Loss -----------------------------------------------------------
    if loss_config['loss_function'] == 'binary_cross_entropy':
        binary_weight = get_params_values(config['SOLVER'], 'binary_weight', None)
        if binary_weight is not None:
            # binary_weight is ratio of background/foreground, use as pos_weight
            pos_weight = torch.tensor(binary_weight).to(device)
        else:
            pos_weight = torch.tensor(1.0).to(device)  # Default balanced weight
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction).to(device)

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
    elif loss_config['loss_function'] == 'focal_loss':
        gamma = get_params_values(loss_config, "gamma", 1.0)
        num_classes = get_params_values(model_config, 'num_classes', None)
        weights = torch.Tensor(num_classes * [1.0]).to(device)
        if class_weights is not None:
            weights = torch.Tensor(class_weights)
        return FocalLoss(gamma=gamma, alpha= weights, reduction=reduction)
    
    # Masked Focal Loss --------------------------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_focal_loss':
        gamma = get_params_values(loss_config, "gamma", 1.0)
        num_classes = get_params_values(model_config, 'num_classes', None)
        weights = torch.Tensor(num_classes * [1.0]).to(device)
        if class_weights is not None:
            weights = torch.Tensor(class_weights)
        return MaskedFocalLoss(gamma=gamma, alpha= weights, reduction=reduction)
    
    # Lovasz Softmax Loss ------------------------------------------------------
    elif loss_config['loss_function'] == 'lovasz_softmax':
        return DynamicCELovaszLoss(
                    total_epochs=20,
                    switch_fraction=0.7,          # slower ramp to Lovasz
                    ce_weight_start=1.0,
                    ce_weight_end=0.65,           # keep CE stronger at the end
                    lovasz_weight_start=0.1,      # small early Lovasz influence
                    lovasz_weight_end=1.0,
                    class_weights=torch.Tensor(class_weights).to(device),
                    reduction=reduction
                )
        
    # Log-Cosh Dice Loss --------------------------------------------------------
    elif loss_config['loss_function'] == 'log_cosh_dice':
        return LogCoshDiceLoss(smooth=1.0, reduction=reduction, class_weights=torch.Tensor(class_weights).to(device),)
    
    elif loss_config['loss_function'] == 'focal_tversky':
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0, reduction=reduction, class_weights=torch.Tensor(class_weights).to(device),)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_config['loss_function']}")


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

# ---------------------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------

class FocalLoss(nn.Module):
        
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
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
    
# ---------------------------------------------------------------------------------------------------------

class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction=None):
        """
        Masked Focal Loss implementation
        
        Args:
            gamma: focusing parameter (default: 2.0)
            alpha: class weighting factor, can be:
                   - None: no weighting
                   - float: same weight for all classes
                   - list/tensor: per-class weights
            reduction: return mean loss vs per element loss
        """
        super(MaskedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        if isinstance(self.alpha, list):
            self.alpha = torch.Tensor(self.alpha)
            
    def forward(self, logits, ground_truth):
        """
        Args:
            logits: (N, C, H, W) unnormalized logits
            ground_truth: (target, mask) where:
                - target: (N, H, W) class indices
                - mask: (N, H, W) boolean mask
        Returns:
            loss: scalar if mean=True, otherwise tensor of shape (M,)
        """
        target, mask = ground_truth
        
        # Permute logits to (N, H, W, C) then flatten
        N, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1)  # (N, H, W, C)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)  # (N*H*W,)
            logits_flat = logits.reshape(-1, C)  # (N*H*W, C)
            target_flat = target.reshape(-1)  # (N*H*W,)
            
            # Select only masked elements
            masked_logits = logits_flat[mask_flat]  # (M, C)
            masked_target = target_flat[mask_flat]  # (M,)
        else:
            masked_logits = logits.reshape(-1, C)  # (N*H*W, C)
            masked_target = target.reshape(-1)  # (N*H*W,)
        
        # Compute focal loss components
        log_probs = F.log_softmax(masked_logits, dim=1)  # (M, C)
        probs = torch.exp(log_probs)  # (M, C)
        
        # Get probabilities and log probabilities for target classes
        log_p_t = log_probs[torch.arange(len(masked_target)), masked_target]  # (M,)
        p_t = probs[torch.arange(len(masked_target)), masked_target]  # (M,)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma  # (M,)
        
        # Compute loss
        loss = -log_p_t * focal_weight  # (M,)
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[masked_target]  # (M,)
            loss *= alpha_t
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss  # (M,) - consistent with FocalLoss


# ---------------------------------------------------------------------------------------------------------

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels, classes='present'):
    if probs.numel() == 0:
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = range(C) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        prob = probs[:, c]
        errors = (fg - prob).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return torch.tensor(0., device=probs.device)
    return torch.mean(torch.stack(losses))

def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 4:
        probas = probas.permute(0, 2, 3, 1).contiguous()
        probas = probas.view(-1, probas.size(-1))
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero(as_tuple=False).squeeze()]
    vlabels = labels[valid.nonzero(as_tuple=False).squeeze()]
    return vprobas, vlabels

class DynamicCELovaszLoss(torch.nn.Module):
    def __init__(self, 
                 total_epochs, 
                 switch_fraction=0.5, 
                 ce_weight_start=1.0, 
                 ce_weight_end=0.5,
                 lovasz_weight_start=0.0, 
                 lovasz_weight_end=1.0,
                 class_weights=None,
                 reduction='mean'):
        """
        total_epochs: total number of training epochs
        switch_fraction: fraction of total epochs over which Lovasz weight ramps up
        ce_weight_start/end: CE weight at start/end of ramp
        lovasz_weight_start/end: Lovasz weight at start/end of ramp
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.switch_fraction = switch_fraction
        self.ce_weight_start = ce_weight_start
        self.ce_weight_end = ce_weight_end
        self.lovasz_weight_start = lovasz_weight_start
        self.lovasz_weight_end = lovasz_weight_end
        self.class_weights = class_weights
        self.reduction = reduction

    def _get_weights(self, epoch):
        ramp_epochs = int(self.total_epochs * self.switch_fraction)
        if epoch >= ramp_epochs:
            return self.ce_weight_end, self.lovasz_weight_end
        t = epoch / ramp_epochs  # 0 → 1 during ramp
        ce_w = self.ce_weight_start + t * (self.ce_weight_end - self.ce_weight_start)
        lovasz_w = self.lovasz_weight_start + t * (self.lovasz_weight_end - self.lovasz_weight_start)
        return ce_w, lovasz_w

    def forward(self, inputs, targets, epoch):
        ce_w, lovasz_w = self._get_weights(epoch)

        ce = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)

        probs = F.softmax(inputs, dim=1)
        probs_flat, labels_flat = flatten_probas(probs, targets)
        lovasz = lovasz_softmax_flat(probs_flat, labels_flat, classes='present')

        return ce_w * ce + lovasz_w * lovasz
    
# ---------------------------------------------------------------------------------------------------------
    
class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, class_weights=None, reduction='mean'):
        """
        smooth: smoothing constant for dice
        class_weights: tensor of shape (num_classes,) for weighting
        reduction: ignored, always returns mean
        """
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets, epoch):
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Dice part
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score
        log_cosh_dice = torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2.0)

        # Apply weights to Dice if given
        if self.class_weights is not None:
            log_cosh_dice = log_cosh_dice * self.class_weights

        # CE part (with weights)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')

        # Decay CE weight linearly
        ce_weight = max(0.0, 1.0 - (epoch / 10.0))

        # Combine losses (always mean)
        total_loss = log_cosh_dice.mean() * (1 - ce_weight) + ce_loss * ce_weight

        return total_loss

# ---------------------------------------------------------------------------------------------------------        
        
class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, class_weights=None, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets, epoch):
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        TP = (probs * targets_one_hot).sum(dims)
        FP = ((1 - targets_one_hot) * probs).sum(dims)
        FN = (targets_one_hot * (1 - probs)).sum(dims)

        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        focal_tversky = (1 - tversky) ** self.gamma

        if self.class_weights is not None:
            focal_tversky = focal_tversky * self.class_weights

        # CE part (with weights)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')

        # Decay CE weight linearly
        ce_weight = max(0.0, 1.0 - (epoch / 10.0))

        # Combine losses (always mean)
        total_loss = focal_tversky.mean() * (1 - ce_weight) + ce_loss * ce_weight

        return total_loss