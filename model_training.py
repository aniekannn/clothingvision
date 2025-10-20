"""
Model Training Utilities for Fashion Recognition System
Includes dataset management, model training, and evaluation functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fashion_recognition_system import FashionRecognitionModel, ClothingCategory

logger = logging.getLogger(__name__)

class FashionDataset(Dataset):
    """Custom dataset for fashion recognition"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()
        
        # Create category and brand mappings
        self.category_to_idx = {cat.value: idx for idx, cat in enumerate(ClothingCategory)}
        self.brand_to_idx = self._create_brand_mapping()
        
    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        metadata_path = self.data_dir / f"{self.split}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_samples(self) -> List[Dict]:
        """Load sample information"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if split_dir.exists():
            for category_dir in split_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    for img_file in category_dir.glob("*.jpg"):
                        sample = {
                            'image_path': str(img_file),
                            'category': category,
                            'brand': self._extract_brand_from_filename(img_file.name)
                        }
                        samples.append(sample)
        
        return samples
    
    def _extract_brand_from_filename(self, filename: str) -> str:
        """Extract brand information from filename"""
        # This would be customized based on your dataset structure
        # For now, we'll use a simple heuristic
        filename_lower = filename.lower()
        
        brand_keywords = {
            'nike': 'Nike',
            'adidas': 'Adidas',
            'gucci': 'Gucci',
            'prada': 'Prada',
            'chanel': 'Chanel',
            'lv': 'Louis Vuitton',
            'louis_vuitton': 'Louis Vuitton'
        }
        
        for keyword, brand in brand_keywords.items():
            if keyword in filename_lower:
                return brand
        
        return 'Unknown'
    
    def _create_brand_mapping(self) -> Dict[str, int]:
        """Create brand to index mapping"""
        brands = set()
        for sample in self.samples:
            brands.add(sample['brand'])
        
        return {brand: idx for idx, brand in enumerate(sorted(brands))}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Get labels
        category_idx = self.category_to_idx.get(sample['category'], 0)
        brand_idx = self.brand_to_idx.get(sample['brand'], 0)
        
        # Create attribute labels (simplified)
        attributes = self._create_attribute_labels(image, sample)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'category': category_idx,
            'brand': brand_idx,
            'attributes': attributes
        }
    
    def _load_image(self, path: str):
        """Load image from path"""
        from PIL import Image
        return Image.open(path).convert('RGB')
    
    def _create_attribute_labels(self, image, sample: Dict) -> torch.Tensor:
        """Create attribute labels (simplified implementation)"""
        # In a real implementation, this would analyze the image
        # For now, we'll return random labels
        num_attributes = 20
        return torch.randint(0, 2, (num_attributes,)).float()

class FashionTrainer:
    """Trainer class for fashion recognition models"""
    
    def __init__(self, model: FashionRecognitionModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss functions
        self.category_criterion = nn.CrossEntropyLoss()
        self.brand_criterion = nn.CrossEntropyLoss()
        self.attribute_criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_history = {
            'category_loss': [],
            'brand_loss': [],
            'attribute_loss': [],
            'total_loss': [],
            'category_acc': [],
            'brand_acc': []
        }
        
        self.val_history = {
            'category_loss': [],
            'brand_loss': [],
            'attribute_loss': [],
            'total_loss': [],
            'category_acc': [],
            'brand_acc': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        category_correct = 0
        brand_correct = 0
        total_samples = 0
        
        category_losses = []
        brand_losses = []
        attribute_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            categories = batch['category'].to(self.device)
            brands = batch['brand'].to(self.device)
            attributes = batch['attributes'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            category_logits, brand_logits, attribute_logits = self.model(images)
            
            # Calculate losses
            category_loss = self.category_criterion(category_logits, categories)
            brand_loss = self.brand_criterion(brand_logits, brands)
            attribute_loss = self.attribute_criterion(attribute_logits, attributes)
            
            total_loss_batch = category_loss + brand_loss + 0.5 * attribute_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            category_losses.append(category_loss.item())
            brand_losses.append(brand_loss.item())
            attribute_losses.append(attribute_loss.item())
            
            # Accuracy calculations
            _, predicted_categories = torch.max(category_logits, 1)
            _, predicted_brands = torch.max(brand_logits, 1)
            
            category_correct += (predicted_categories == categories).sum().item()
            brand_correct += (predicted_brands == brands).sum().item()
            total_samples += categories.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(dataloader)}, '
                           f'Loss: {total_loss_batch.item():.4f}')
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        category_acc = category_correct / total_samples
        brand_acc = brand_correct / total_samples
        
        epoch_stats = {
            'total_loss': avg_loss,
            'category_loss': np.mean(category_losses),
            'brand_loss': np.mean(brand_losses),
            'attribute_loss': np.mean(attribute_losses),
            'category_acc': category_acc,
            'brand_acc': brand_acc
        }
        
        return epoch_stats
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        category_correct = 0
        brand_correct = 0
        total_samples = 0
        
        category_losses = []
        brand_losses = []
        attribute_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                categories = batch['category'].to(self.device)
                brands = batch['brand'].to(self.device)
                attributes = batch['attributes'].to(self.device)
                
                # Forward pass
                category_logits, brand_logits, attribute_logits = self.model(images)
                
                # Calculate losses
                category_loss = self.category_criterion(category_logits, categories)
                brand_loss = self.brand_criterion(brand_logits, brands)
                attribute_loss = self.attribute_criterion(attribute_logits, attributes)
                
                total_loss_batch = category_loss + brand_loss + 0.5 * attribute_loss
                total_loss += total_loss_batch.item()
                
                category_losses.append(category_loss.item())
                brand_losses.append(brand_loss.item())
                attribute_losses.append(attribute_loss.item())
                
                # Accuracy calculations
                _, predicted_categories = torch.max(category_logits, 1)
                _, predicted_brands = torch.max(brand_logits, 1)
                
                category_correct += (predicted_categories == categories).sum().item()
                brand_correct += (predicted_brands == brands).sum().item()
                total_samples += categories.size(0)
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        category_acc = category_correct / total_samples
        brand_acc = brand_correct / total_samples
        
        epoch_stats = {
            'total_loss': avg_loss,
            'category_loss': np.mean(category_losses),
            'brand_loss': np.mean(brand_losses),
            'attribute_loss': np.mean(attribute_losses),
            'category_acc': category_acc,
            'brand_acc': brand_acc
        }
        
        return epoch_stats
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, save_path: str = "best_model.pth"):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_stats = self.train_epoch(train_loader)
            self.train_history['total_loss'].append(train_stats['total_loss'])
            self.train_history['category_loss'].append(train_stats['category_loss'])
            self.train_history['brand_loss'].append(train_stats['brand_loss'])
            self.train_history['attribute_loss'].append(train_stats['attribute_loss'])
            self.train_history['category_acc'].append(train_stats['category_acc'])
            self.train_history['brand_acc'].append(train_stats['brand_acc'])
            
            # Validate
            val_stats = self.validate_epoch(val_loader)
            self.val_history['total_loss'].append(val_stats['total_loss'])
            self.val_history['category_loss'].append(val_stats['category_loss'])
            self.val_history['brand_loss'].append(val_stats['brand_loss'])
            self.val_history['attribute_loss'].append(val_stats['attribute_loss'])
            self.val_history['category_acc'].append(val_stats['category_acc'])
            self.val_history['brand_acc'].append(val_stats['brand_acc'])
            
            # Update scheduler
            self.scheduler.step(val_stats['total_loss'])
            
            # Save best model
            if val_stats['total_loss'] < best_val_loss:
                best_val_loss = val_stats['total_loss']
                torch.save(self.model.state_dict(), save_path)
                logger.info(f'New best model saved with validation loss: {best_val_loss:.4f}')
            
            # Log epoch statistics
            logger.info(f'Train - Loss: {train_stats["total_loss"]:.4f}, '
                       f'Category Acc: {train_stats["category_acc"]:.4f}, '
                       f'Brand Acc: {train_stats["brand_acc"]:.4f}')
            logger.info(f'Val - Loss: {val_stats["total_loss"]:.4f}, '
                       f'Category Acc: {val_stats["category_acc"]:.4f}, '
                       f'Brand Acc: {val_stats["brand_acc"]:.4f}')
        
        logger.info('Training completed!')
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.train_history['total_loss'], label='Train')
        axes[0, 0].plot(self.val_history['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        
        axes[0, 1].plot(self.train_history['category_loss'], label='Train')
        axes[0, 1].plot(self.val_history['category_loss'], label='Validation')
        axes[0, 1].set_title('Category Loss')
        axes[0, 1].legend()
        
        axes[0, 2].plot(self.train_history['brand_loss'], label='Train')
        axes[0, 2].plot(self.val_history['brand_loss'], label='Validation')
        axes[0, 2].set_title('Brand Loss')
        axes[0, 2].legend()
        
        # Accuracy plots
        axes[1, 0].plot(self.train_history['category_acc'], label='Train')
        axes[1, 0].plot(self.val_history['category_acc'], label='Validation')
        axes[1, 0].set_title('Category Accuracy')
        axes[1, 0].legend()
        
        axes[1, 1].plot(self.train_history['brand_acc'], label='Train')
        axes[1, 1].plot(self.val_history['brand_acc'], label='Validation')
        axes[1, 1].set_title('Brand Accuracy')
        axes[1, 1].legend()
        
        # Attribute loss
        axes[1, 2].plot(self.train_history['attribute_loss'], label='Train')
        axes[1, 2].plot(self.val_history['attribute_loss'], label='Validation')
        axes[1, 2].set_title('Attribute Loss')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, model: FashionRecognitionModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        all_category_preds = []
        all_category_targets = []
        all_brand_preds = []
        all_brand_targets = []
        
        total_loss = 0
        
        category_criterion = nn.CrossEntropyLoss()
        brand_criterion = nn.CrossEntropyLoss()
        attribute_criterion = nn.BCELoss()
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                categories = batch['category'].to(self.device)
                brands = batch['brand'].to(self.device)
                attributes = batch['attributes'].to(self.device)
                
                # Forward pass
                category_logits, brand_logits, attribute_logits = self.model(images)
                
                # Calculate loss
                category_loss = category_criterion(category_logits, categories)
                brand_loss = brand_criterion(brand_logits, brands)
                attribute_loss = attribute_criterion(attribute_logits, attributes)
                
                total_loss += (category_loss + brand_loss + 0.5 * attribute_loss).item()
                
                # Get predictions
                _, category_preds = torch.max(category_logits, 1)
                _, brand_preds = torch.max(brand_logits, 1)
                
                all_category_preds.extend(category_preds.cpu().numpy())
                all_category_targets.extend(categories.cpu().numpy())
                all_brand_preds.extend(brand_preds.cpu().numpy())
                all_brand_targets.extend(brands.cpu().numpy())
        
        # Calculate metrics
        category_acc = np.mean(np.array(all_category_preds) == np.array(all_category_targets))
        brand_acc = np.mean(np.array(all_brand_preds) == np.array(all_brand_targets))
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'category_accuracy': category_acc,
            'brand_accuracy': brand_acc,
            'category_predictions': all_category_preds,
            'category_targets': all_category_targets,
            'brand_predictions': all_brand_preds,
            'brand_targets': all_brand_targets
        }
    
    def generate_classification_report(self, dataloader: DataLoader, 
                                     category_names: List[str], 
                                     brand_names: List[str]) -> Dict[str, str]:
        """Generate detailed classification reports"""
        eval_results = self.evaluate_model(dataloader)
        
        # Category report
        category_report = classification_report(
            eval_results['category_targets'],
            eval_results['category_predictions'],
            target_names=category_names,
            zero_division=0
        )
        
        # Brand report
        brand_report = classification_report(
            eval_results['brand_targets'],
            eval_results['brand_predictions'],
            target_names=brand_names,
            zero_division=0
        )
        
        return {
            'category_report': category_report,
            'brand_report': brand_report
        }
    
    def plot_confusion_matrices(self, dataloader: DataLoader,
                              category_names: List[str],
                              brand_names: List[str],
                              save_path: str = "confusion_matrices.png"):
        """Plot confusion matrices for categories and brands"""
        eval_results = self.evaluate_model(dataloader)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Category confusion matrix
        cm_category = confusion_matrix(
            eval_results['category_targets'],
            eval_results['category_predictions']
        )
        sns.heatmap(cm_category, annot=True, fmt='d', cmap='Blues',
                   xticklabels=category_names, yticklabels=category_names, ax=ax1)
        ax1.set_title('Category Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Brand confusion matrix
        cm_brand = confusion_matrix(
            eval_results['brand_targets'],
            eval_results['brand_predictions']
        )
        sns.heatmap(cm_brand, annot=True, fmt='d', cmap='Greens',
                   xticklabels=brand_names, yticklabels=brand_names, ax=ax2)
        ax2.set_title('Brand Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_data_transforms():
    """Create data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    """Main training function"""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets (you would need to provide actual data paths)
    data_dir = "path/to/your/fashion/dataset"
    
    # For demonstration, we'll create dummy datasets
    # In practice, you would use:
    # train_dataset = FashionDataset(data_dir, 'train', train_transform)
    # val_dataset = FashionDataset(data_dir, 'val', val_transform)
    
    # Create model
    model = FashionRecognitionModel(num_categories=15, num_brands=50)
    
    # Create trainer
    trainer = FashionTrainer(model, device)
    
    # Create data loaders (you would need actual datasets)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Train model
    # trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Plot training history
    # trainer.plot_training_history()
    
    logger.info("Training setup completed. Provide actual dataset paths to begin training.")

if __name__ == "__main__":
    main()
