"""
MultiTask-ResNet18 Vision Model
Joint Classification and Segmentation using shared ResNet18 backbone
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ClassificationHead(nn.Module):
    """Classification head with global average pooling and FC layers"""
    
    def __init__(self, in_features=512, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SegmentationHead(nn.Module):
    """Segmentation head with transposed convolution decoder"""
    
    def __init__(self, in_features=512, num_classes=10):
        super(SegmentationHead, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_features, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        x = self.decoder(x)
        return x


class MultiTaskResNet18(nn.Module):
    """
    Multi-Task ResNet18 for joint Classification and Segmentation
    
    Args:
        num_classes_cls (int): Number of classification classes
        num_classes_seg (int): Number of segmentation classes
        pretrained (bool): Whether to use pretrained ResNet18 weights
    """
    
    def __init__(self, num_classes_cls=10, num_classes_seg=10, pretrained=True):
        super(MultiTaskResNet18, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract backbone (remove classification layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone_out_channels = 512
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            in_features=self.backbone_out_channels,
            num_classes=num_classes_cls
        )
        
        self.segmentation_head = SegmentationHead(
            in_features=self.backbone_out_channels,
            num_classes=num_classes_seg
        )
    
    def forward(self, x):
        """
        Forward pass for multi-task learning
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
        
        Returns:
            tuple: (classification_logits, segmentation_logits)
        """
        # Shared backbone
        backbone_features = self.backbone(x)
        
        # Task-specific predictions
        cls_out = self.classification_head(backbone_features)
        seg_out = self.segmentation_head(backbone_features)
        
        return cls_out, seg_out


if __name__ == "__main__":
    # Test model
    model = MultiTaskResNet18(num_classes_cls=10, num_classes_seg=10)
    x = torch.randn(4, 3, 224, 224)
    cls_out, seg_out = model(x)
    
    print(f"Classification output shape: {cls_out.shape}")  # [4, 10]
    print(f"Segmentation output shape: {seg_out.shape}")    # [4, 10, H, W]
