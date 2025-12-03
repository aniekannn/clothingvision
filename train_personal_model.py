#!/usr/bin/env python3
"""
Training Script for Personal Clothing Recognition Model
Optimized for improving camera accuracy with your personal dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import numpy as np
import json
import os
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import time

# Import your existing modules
from model_training import FashionTrainer, FashionDataset, create_data_transforms
from fashion_recognition_system import FashionRecognitionModel, ClothingCategory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalFashionTrainer(FashionTrainer):
    """Enhanced trainer specifically for personal clothing datasets"""
    
    def __init__(self, model, device, personal_dataset_path):
        super().__init__(model, device)
        self.personal_dataset_path = personal_dataset_path
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.max_patience = 10
        
        # Enhanced optimizer for better convergence
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    def train_with_early_stopping(self, train_loader, val_loader, num_epochs=100):
        """Train with early stopping and best model saving"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                       f"Category Acc: {train_metrics['category_acc']:.4f}, "
                       f"Brand Acc: {train_metrics['brand_acc']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                       f"Category Acc: {val_metrics['category_acc']:.4f}, "
                       f"Brand Acc: {val_metrics['brand_acc']:.4f}")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            val_acc = val_metrics['category_acc']
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # Save best model
                self.save_model(f"best_personal_model_epoch_{epoch+1}.pth")
                logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Update history
            for key in train_metrics:
                self.train_history[key].append(train_metrics[key])
            for key in val_metrics:
                self.val_history[key].append(val_metrics[key])
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Plot training history
        self.plot_training_history("personal_model_training_history.png")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        torch.save(checkpoint, filename)
        logger.info(f"Model saved as {filename}")

def create_enhanced_transforms():
    """Create enhanced data transforms for better camera accuracy"""
    
    # Training transforms with more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_personal_dataset(data_dir, split, transform):
    """Create dataset from personal clothing photos"""
    dataset = FashionDataset(data_dir, split, transform)
    return dataset

def analyze_dataset_distribution(dataset):
    """Analyze dataset distribution for better training"""
    categories = [sample['category'] for sample in dataset.samples]
    brands = [sample['brand'] for sample in dataset.samples]
    
    category_counts = Counter(categories)
    brand_counts = Counter(brands)
    
    logger.info("Dataset Distribution Analysis:")
    logger.info(f"Total samples: {len(dataset.samples)}")
    logger.info(f"Categories: {dict(category_counts)}")
    logger.info(f"Brands: {dict(brand_counts)}")
    
    # Check for class imbalance
    min_category_count = min(category_counts.values())
    max_category_count = max(category_counts.values())
    imbalance_ratio = max_category_count / min_category_count if min_category_count > 0 else float('inf')
    
    if imbalance_ratio > 3:
        logger.warning(f"High class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        logger.warning("Consider using weighted sampling or data augmentation")
    
    return category_counts, brand_counts

def create_weighted_sampler(dataset):
    """Create weighted sampler to handle class imbalance"""
    categories = [sample['category'] for sample in dataset.samples]
    category_counts = Counter(categories)
    
    # Calculate weights
    weights = []
    for sample in dataset.samples:
        weight = 1.0 / category_counts[sample['category']]
        weights.append(weight)
    
    # Create weighted sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler

def evaluate_model_accuracy(model, dataloader, device, class_names=None):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_category_preds = []
    all_category_targets = []
    all_brand_preds = []
    all_brand_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            categories = batch['category'].to(device)
            brands = batch['brand'].to(device)
            
            category_logits, brand_logits, _ = model(images)
            
            _, category_preds = torch.max(category_logits, 1)
            _, brand_preds = torch.max(brand_logits, 1)
            
            all_category_preds.extend(category_preds.cpu().numpy())
            all_category_targets.extend(categories.cpu().numpy())
            all_brand_preds.extend(brand_preds.cpu().numpy())
            all_brand_targets.extend(brands.cpu().numpy())
    
    # Calculate accuracies
    category_acc = np.mean(np.array(all_category_preds) == np.array(all_category_targets))
    brand_acc = np.mean(np.array(all_brand_preds) == np.array(all_brand_targets))
    
    logger.info(f"Category Accuracy: {category_acc:.4f}")
    logger.info(f"Brand Accuracy: {brand_acc:.4f}")
    
    # Generate classification report
    if class_names:
        category_report = classification_report(
            all_category_targets, all_category_preds, 
            target_names=class_names, output_dict=True
        )
        
        logger.info("Category Classification Report:")
        for class_name, metrics in category_report.items():
            if isinstance(metrics, dict):
                logger.info(f"{class_name}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    return {
        'category_accuracy': category_acc,
        'brand_accuracy': brand_acc,
        'category_predictions': all_category_preds,
        'category_targets': all_category_targets,
        'brand_predictions': all_brand_preds,
        'brand_targets': all_brand_targets
    }

def main():
    """Main training function for personal clothing recognition"""
    
    # Configuration
    PERSONAL_DATASET_PATH = "my_personal_clothing_dataset"
    BATCH_SIZE = 16  # Smaller batch size for personal dataset
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if personal dataset exists
    if not os.path.exists(PERSONAL_DATASET_PATH):
        logger.error(f"Personal dataset not found at {PERSONAL_DATASET_PATH}")
        logger.error("Please run add_personal_dataset.py first to create your dataset")
        return
    
    # Create enhanced transforms
    train_transform, val_transform = create_enhanced_transforms()
    
    # Create datasets
    logger.info("Loading personal dataset...")
    train_dataset = create_personal_dataset(PERSONAL_DATASET_PATH, 'train', train_transform)
    val_dataset = create_personal_dataset(PERSONAL_DATASET_PATH, 'val', val_transform)
    test_dataset = create_personal_dataset(PERSONAL_DATASET_PATH, 'test', val_transform)
    
    # Analyze dataset distribution
    train_category_counts, train_brand_counts = analyze_dataset_distribution(train_dataset)
    
    # Create weighted sampler for imbalanced data
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    num_categories = len(ClothingCategory)
    num_brands = len(train_dataset.brand_to_idx)
    
    logger.info(f"Creating model with {num_categories} categories and {num_brands} brands")
    
    # Use a more powerful backbone for better accuracy
    model = FashionRecognitionModel(
        num_categories=num_categories, 
        num_brands=num_brands,
        backbone='resnet50'  # Use ResNet50 for better accuracy
    )
    
    # Create trainer
    trainer = PersonalFashionTrainer(model, device, PERSONAL_DATASET_PATH)
    
    # Train model
    logger.info("Starting training...")
    trainer.train_with_early_stopping(train_loader, val_loader, NUM_EPOCHS)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = evaluate_model_accuracy(
        model, test_loader, device, 
        class_names=list(ClothingCategory)
    )
    
    # Save final model
    trainer.save_model("final_personal_model.pth")
    
    logger.info("Training completed!")
    logger.info(f"Final test accuracy: {test_results['category_accuracy']:.4f}")
    
    return model, test_results

if __name__ == "__main__":
    model, results = main()

