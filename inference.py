"""
Inference script for MultiTask-ResNet18-Vision
"""

import torch
import argparse
from PIL import Image
from torchvision import transforms
import numpy as np

from src.model import MultiTaskResNet18
from src.utils import load_checkpoint


def inference(model_path, image_path, device='cpu', num_classes_cls=10, num_classes_seg=10):
    """
    Run inference on a single image
    
    Args:
        model_path: Path to model checkpoint
        image_path: Path to input image
        device: Device to use
        num_classes_cls: Number of classification classes
        num_classes_seg: Number of segmentation classes
    
    Returns:
        Dictionary with classification and segmentation outputs
    """
    # Load model
    model = MultiTaskResNet18(num_classes_cls=num_classes_cls, 
                              num_classes_seg=num_classes_seg, 
                              pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        cls_out, seg_out = model(image_tensor)
    
    # Process outputs
    cls_probs = torch.softmax(cls_out, dim=1)
    cls_pred = torch.argmax(cls_probs, dim=1)
    cls_confidence = torch.max(cls_probs, dim=1).values
    
    seg_pred = torch.argmax(seg_out, dim=1)
    
    return {
        'classification': {
            'class': cls_pred.item(),
            'confidence': cls_confidence.item(),
            'probabilities': cls_probs.cpu().numpy()
        },
        'segmentation': {
            'mask': seg_pred.cpu().squeeze().numpy(),
            'logits': seg_out.cpu().squeeze().numpy()
        }
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    results = inference(
        model_path=args.model,
        image_path=args.image,
        device=device,
        num_classes_cls=args.num_classes_cls,
        num_classes_seg=args.num_classes_seg
    )
    
    print(f"\nClassification Results:")
    print(f"  Predicted Class: {results['classification']['class']}")
    print(f"  Confidence: {results['classification']['confidence']:.4f}")
    
    print(f"\nSegmentation Results:")
    print(f"  Mask Shape: {results['segmentation']['mask'].shape}")
    print(f"  Unique Classes: {np.unique(results['segmentation']['mask'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with MultiTask-ResNet18')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--num-classes-cls', type=int, default=10,
                       help='Number of classification classes')
    parser.add_argument('--num-classes-seg', type=int, default=10,
                       help='Number of segmentation classes')
    parser.add_argument('--use-cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    main(args)
