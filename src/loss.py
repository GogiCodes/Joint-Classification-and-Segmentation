"""
Loss functions for multi-task learning
Composite loss: Cross-Entropy for classification + Dice Loss for segmentation
"""

import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    """Dice coefficient loss for segmentation"""
    
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] integer labels
        
        Returns:
            Dice loss (float)
        """
        # Get class predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Flatten
        predictions = predictions.permute(0, 2, 3, 1).contiguous().view(-1, predictions.size(1))
        targets = targets.view(-1)
        
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=predictions.size(1)).float()
        
        # Dice coefficient
        intersection = (predictions * targets_onehot).sum(dim=0)
        union = predictions.sum(dim=0) + targets_onehot.sum(dim=0)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - dice (loss)
        loss = 1 - dice.mean()
        
        return loss


class MultiTaskLoss(torch.nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Args:
            alpha (float): Weight for classification loss
            beta (float): Weight for segmentation loss
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, cls_out, seg_out, cls_labels, seg_masks):
        """
        Args:
            cls_out: [B, num_classes] classification logits
            seg_out: [B, num_classes, H, W] segmentation logits
            cls_labels: [B] classification labels
            seg_masks: [B, H, W] segmentation masks
        
        Returns:
            Loss dictionary with total and individual losses
        """
        # Classification loss (Cross-Entropy)
        ce_loss = self.ce_loss(cls_out, cls_labels)
        
        # Segmentation loss (Dice)
        dice_loss = self.dice_loss(seg_out, seg_masks)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + self.beta * dice_loss
        
        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss
        }
