"""
Training script for MultiTask-ResNet18-Vision
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.model import MultiTaskResNet18
from src.loss import MultiTaskLoss
from src.data import get_dataloaders
from src.utils import (
    calculate_accuracy, calculate_iou, MetricsTracker,
    save_checkpoint, load_checkpoint
)


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """Train for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        cls_labels = batch['cls_label'].to(device)
        seg_masks = batch['seg_mask'].to(device)
        
        # Forward pass
        cls_out, seg_out = model(images)
        
        # Compute loss
        loss_dict = loss_fn(cls_out, seg_out, cls_labels, seg_masks)
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        cls_acc = calculate_accuracy(cls_out, cls_labels)
        _, miou = calculate_iou(seg_out, seg_masks, num_classes=seg_out.size(1))
        
        metrics.update(loss_dict, cls_acc, miou)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] - {metrics}")
    
    return metrics.get_averages()


def val_epoch(model, val_loader, loss_fn, device):
    """Validate for one epoch"""
    model.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            cls_labels = batch['cls_label'].to(device)
            seg_masks = batch['seg_mask'].to(device)
            
            # Forward pass
            cls_out, seg_out = model(images)
            
            # Compute loss
            loss_dict = loss_fn(cls_out, seg_out, cls_labels, seg_masks)
            
            # Compute metrics
            cls_acc = calculate_accuracy(cls_out, cls_labels)
            _, miou = calculate_iou(seg_out, seg_masks, num_classes=seg_out.size(1))
            
            metrics.update(loss_dict, cls_acc, miou)
    
    return metrics.get_averages()


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = MultiTaskResNet18(
        num_classes_cls=args.num_classes_cls,
        num_classes_seg=args.num_classes_seg,
        pretrained=args.pretrained
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create loss function
    loss_fn = MultiTaskLoss(alpha=args.alpha, beta=args.beta)
    
    # Create dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_miou = 0
    start_epoch = 0
    
    if args.resume and os.path.exists(args.resume):
        model, optimizer, start_epoch, best_miou = load_checkpoint(
            model, optimizer, args.resume
        )
        start_epoch += 1
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_metrics = val_epoch(model, val_loader, loss_fn, device)
        
        # Log metrics
        print(f"\nTrain - {MetricsTracker.__str__(lambda: train_metrics)}")
        print(f"Val   - {MetricsTracker.__str__(lambda: val_metrics)}")
        
        for key, val in train_metrics.items():
            writer.add_scalar(f"train/{key}", val, epoch)
        for key, val in val_metrics.items():
            writer.add_scalar(f"val/{key}", val, epoch)
        
        # Save best checkpoint
        if val_metrics['avg_miou'] > best_miou:
            best_miou = val_metrics['avg_miou']
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_miou, checkpoint_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, best_miou, checkpoint_path)
        
        # Update learning rate
        scheduler.step()
    
    writer.close()
    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MultiTask-ResNet18 model')
    
    # Model arguments
    parser.add_argument('--num-classes-cls', type=int, default=10,
                       help='Number of classification classes')
    parser.add_argument('--num-classes-seg', type=int, default=10,
                       help='Number of segmentation classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ResNet18')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for segmentation loss')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Logging/checkpoint arguments
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint-dir', type=str, default='./models',
                       help='Directory for saving checkpoints')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to resume from checkpoint')
    
    args = parser.parse_args()
    
    main(args)
