"""
Utility functions for evaluation and metrics
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_iou(predictions, targets, num_classes):
    """
    Calculate Intersection over Union (IoU) for segmentation
    
    Args:
        predictions: [B, C, H, W] model predictions
        targets: [B, H, W] ground truth masks
        num_classes: number of classes
    
    Returns:
        per_class_iou: IoU for each class
        miou: mean IoU
    """
    predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    iou_scores = []
    
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx)
        target_mask = (targets == class_idx)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_scores.append(iou)
    
    miou = np.mean(iou_scores)
    return iou_scores, miou


def calculate_accuracy(predictions, targets):
    """
    Calculate classification accuracy
    
    Args:
        predictions: [B, num_classes] logits or probabilities
        targets: [B] ground truth labels
    
    Returns:
        accuracy: classification accuracy
    """
    predictions = torch.argmax(predictions, dim=1)
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    accuracy = accuracy_score(targets, predictions)
    return accuracy


class MetricsTracker:
    """Track metrics during training/validation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cls_losses = []
        self.seg_losses = []
        self.total_losses = []
        self.accuracies = []
        self.mious = []
    
    def update(self, loss_dict, cls_acc, miou):
        """
        Update metrics
        
        Args:
            loss_dict: dictionary with 'ce_loss', 'dice_loss', 'total'
            cls_acc: classification accuracy
            miou: mean IoU
        """
        self.cls_losses.append(loss_dict['ce_loss'].item())
        self.seg_losses.append(loss_dict['dice_loss'].item())
        self.total_losses.append(loss_dict['total'].item())
        self.accuracies.append(cls_acc)
        self.mious.append(miou)
    
    def get_averages(self):
        """Get average metrics"""
        return {
            'avg_ce_loss': np.mean(self.cls_losses) if self.cls_losses else 0,
            'avg_dice_loss': np.mean(self.seg_losses) if self.seg_losses else 0,
            'avg_total_loss': np.mean(self.total_losses) if self.total_losses else 0,
            'avg_accuracy': np.mean(self.accuracies) if self.accuracies else 0,
            'avg_miou': np.mean(self.mious) if self.mious else 0
        }
    
    def __str__(self):
        avgs = self.get_averages()
        return (f"CE Loss: {avgs['avg_ce_loss']:.4f} | "
                f"Dice Loss: {avgs['avg_dice_loss']:.4f} | "
                f"Total Loss: {avgs['avg_total_loss']:.4f} | "
                f"Accuracy: {avgs['avg_accuracy']:.4f} | "
                f"mIoU: {avgs['avg_miou']:.4f}")


def save_checkpoint(model, optimizer, epoch, best_metric, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    return model, optimizer, epoch, best_metric
