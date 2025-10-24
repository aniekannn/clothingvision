"""
Dataset Management Utilities for Fashion Recognition System
Handles dataset creation, preprocessing, and augmentation
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import requests
from urllib.parse import urlparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import tarfile
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Types of datasets"""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

@dataclass
class ImageAnnotation:
    """Data class for image annotations"""
    image_path: str
    category: str
    brand: str
    attributes: List[str]
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    metadata: Optional[Dict] = None

@dataclass
class DatasetStats:
    """Dataset statistics"""
    total_images: int
    categories: Dict[str, int]
    brands: Dict[str, int]
    attributes: Dict[str, int]
    avg_image_size: Tuple[int, int]
    file_size_mb: float

class FashionDatasetManager:
    """Manager for fashion datasets"""
    
    def __init__(self, dataset_root: str = "fashion_dataset"):
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(exist_ok=True)
        
        # Create dataset structure
        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "val"
        self.test_dir = self.dataset_root / "test"
        self.annotations_dir = self.dataset_root / "annotations"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir, self.annotations_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load existing annotations
        self.annotations = self.load_annotations()
        
        # Category and brand mappings
        self.categories = [
            'shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'bag', 
            'watch', 'belt', 'tie', 'socks', 'underwear', 'swimwear', 
            'coat', 'sweater', 'hoodie', 'sweatpants'
        ]
        
        self.brands = [
            'Nike', 'Adidas', 'Gucci', 'Louis Vuitton', 'Chanel', 'Prada',
            'Versace', 'Armani', 'Calvin Klein', 'Tommy Hilfiger', 'Ralph Lauren',
            'Zara', 'H&M', 'Uniqlo', 'Gap', 'Levi\'s', 'Diesel', 'Guess',
            'Coach', 'Michael Kors', 'Kate Spade', 'Fendi', 'Balenciaga',
            'Saint Laurent', 'Burberry', 'Hermès', 'Valentino', 'Dior',
            'Givenchy', 'Bottega Veneta', 'Celine', 'Loewe', 'Jil Sander',
            'Acne Studios', 'Off-White', 'Supreme', 'Bape', 'Stone Island',
            'Moncler', 'Canada Goose', 'North Face', 'Patagonia', 'Columbia',
            'Under Armour', 'New Balance', 'Converse', 'Vans', 'Timberland'
        ]
        
        self.attributes = [
            'red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple',
            'striped', 'polka_dot', 'solid', 'patterned', 'denim', 'leather',
            'cotton', 'silk', 'wool', 'casual', 'formal', 'sporty'
        ]
    
    def load_annotations(self) -> Dict[str, List[ImageAnnotation]]:
        """Load existing annotations"""
        annotations = {'train': [], 'val': [], 'test': []}
        
        for split in annotations.keys():
            annotation_file = self.annotations_dir / f"{split}_annotations.json"
            if annotation_file.exists():
                try:
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        annotations[split] = [ImageAnnotation(**item) for item in data]
                    logger.info(f"Loaded {len(annotations[split])} annotations for {split}")
                except Exception as e:
                    logger.error(f"Error loading annotations for {split}: {e}")
        
        return annotations
    
    def save_annotations(self):
        """Save annotations to files"""
        for split, annotation_list in self.annotations.items():
            annotation_file = self.annotations_dir / f"{split}_annotations.json"
            try:
                with open(annotation_file, 'w') as f:
                    json.dump([asdict(annotation) for annotation in annotation_list], 
                             f, indent=2)
                logger.info(f"Saved {len(annotation_list)} annotations for {split}")
            except Exception as e:
                logger.error(f"Error saving annotations for {split}: {e}")
    
    def add_image(self, image_path: str, category: str, brand: str, 
                  attributes: List[str], split: str = 'train',
                  bbox: Optional[Tuple[int, int, int, int]] = None,
                  metadata: Optional[Dict] = None) -> bool:
        """Add an image to the dataset"""
        try:
            # Validate inputs
            if category not in self.categories:
                logger.warning(f"Unknown category: {category}")
            
            if brand not in self.brands:
                logger.warning(f"Unknown brand: {brand}")
            
            # Create annotation
            annotation = ImageAnnotation(
                image_path=image_path,
                category=category,
                brand=brand,
                attributes=attributes,
                bbox=bbox,
                metadata=metadata
            )
            
            # Add to appropriate split
            if split in self.annotations:
                self.annotations[split].append(annotation)
                logger.info(f"Added image to {split} dataset: {image_path}")
                return True
            else:
                logger.error(f"Invalid split: {split}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding image: {e}")
            return False
    
    def download_image(self, url: str, filename: str, split: str = 'train') -> Optional[str]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
            
            # Create filename
            if not filename.endswith(ext):
                filename += ext
            
            # Save image
            save_path = self.dataset_root / split / filename
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded image: {filename}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def batch_download(self, url_list: List[str], split: str = 'train', 
                      max_workers: int = 5) -> List[Optional[str]]:
        """Download multiple images in parallel"""
        downloaded_paths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            future_to_url = {
                executor.submit(self.download_image, url, f"image_{i}", split): url
                for i, url in enumerate(url_list)
            }
            
            # Collect results
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    downloaded_paths.append(result)
                except Exception as e:
                    logger.error(f"Error downloading {url}: {e}")
                    downloaded_paths.append(None)
        
        return downloaded_paths
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224),
                        normalize: bool = True) -> Optional[np.ndarray]:
        """Preprocess image for training"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, target_size)
            
            # Normalize
            if normalize:
                image = image.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = 'basic') -> List[np.ndarray]:
        """Apply data augmentation to image"""
        augmented_images = [image.copy()]
        
        try:
            if augmentation_type == 'basic':
                # Basic augmentations
                augmented_images.extend(self._basic_augmentations(image))
            elif augmentation_type == 'advanced':
                # Advanced augmentations
                augmented_images.extend(self._advanced_augmentations(image))
            
            return augmented_images
            
        except Exception as e:
            logger.error(f"Error augmenting image: {e}")
            return [image]
    
    def _basic_augmentations(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply basic augmentations"""
        augmented = []
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        # Brightness adjustment
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        enhancer = ImageEnhance.Brightness(pil_image)
        bright = enhancer.enhance(1.2)
        dark = enhancer.enhance(0.8)
        augmented.append(np.array(bright) / 255.0)
        augmented.append(np.array(dark) / 255.0)
        
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(pil_image)
        high_contrast = enhancer.enhance(1.2)
        low_contrast = enhancer.enhance(0.8)
        augmented.append(np.array(high_contrast) / 255.0)
        augmented.append(np.array(low_contrast) / 255.0)
        
        return augmented
    
    def _advanced_augmentations(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply advanced augmentations"""
        augmented = []
        
        # Rotation
        for angle in [15, -15, 30, -30]:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h))
            augmented.append(rotated)
        
        # Translation
        for dx, dy in [(10, 10), (-10, 10), (10, -10), (-10, -10)]:
            h, w = image.shape[:2]
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            translated = cv2.warpAffine(image, matrix, (w, h))
            augmented.append(translated)
        
        # Noise addition
        noise = np.random.normal(0, 0.01, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        augmented.append(noisy)
        
        return augmented
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                     test_ratio: float = 0.1, stratify_by: str = 'category') -> bool:
        """Split dataset into train/val/test"""
        try:
            # Combine all annotations
            all_annotations = []
            for split_annotations in self.annotations.values():
                all_annotations.extend(split_annotations)
            
            if not all_annotations:
                logger.error("No annotations to split")
                return False
            
            # Prepare stratification labels
            if stratify_by == 'category':
                labels = [ann.category for ann in all_annotations]
            elif stratify_by == 'brand':
                labels = [ann.brand for ann in all_annotations]
            else:
                labels = None
            
            # Split annotations
            if labels:
                train_anns, temp_anns = train_test_split(
                    all_annotations, train_size=train_ratio, stratify=labels, random_state=42
                )
                
                val_size = val_ratio / (val_ratio + test_ratio)
                val_anns, test_anns = train_test_split(
                    temp_anns, train_size=val_size, 
                    stratify=[labels[i] for i in range(len(all_annotations)) 
                             if all_annotations[i] in temp_anns], 
                    random_state=42
                )
            else:
                train_anns, temp_anns = train_test_split(
                    all_annotations, train_size=train_ratio, random_state=42
                )
                val_anns, test_anns = train_test_split(
                    temp_anns, train_size=val_ratio / (val_ratio + test_ratio), 
                    random_state=42
                )
            
            # Update annotations
            self.annotations['train'] = train_anns
            self.annotations['val'] = val_anns
            self.annotations['test'] = test_anns
            
            # Save updated annotations
            self.save_annotations()
            
            logger.info(f"Dataset split completed: {len(train_anns)} train, "
                       f"{len(val_anns)} val, {len(test_anns)} test")
            return True
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return False
    
    def get_dataset_stats(self) -> DatasetStats:
        """Get dataset statistics"""
        total_images = sum(len(anns) for anns in self.annotations.values())
        
        # Count categories, brands, and attributes
        categories = Counter()
        brands = Counter()
        attributes = Counter()
        
        image_sizes = []
        total_size_mb = 0
        
        for split_annotations in self.annotations.values():
            for ann in split_annotations:
                categories[ann.category] += 1
                brands[ann.brand] += 1
                
                for attr in ann.attributes:
                    attributes[attr] += 1
                
                # Get image size and file size
                try:
                    image_path = Path(ann.image_path)
                    if image_path.exists():
                        with Image.open(image_path) as img:
                            image_sizes.append(img.size)
                        total_size_mb += image_path.stat().st_size / (1024 * 1024)
                except Exception:
                    pass
        
        # Calculate average image size
        avg_image_size = (0, 0)
        if image_sizes:
            avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
            avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)
            avg_image_size = (int(avg_width), int(avg_height))
        
        return DatasetStats(
            total_images=total_images,
            categories=dict(categories),
            brands=dict(brands),
            attributes=dict(attributes),
            avg_image_size=avg_image_size,
            file_size_mb=total_size_mb
        )
    
    def visualize_dataset_stats(self, save_path: str = "dataset_stats.png"):
        """Visualize dataset statistics"""
        stats = self.get_dataset_stats()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Categories distribution
        categories = list(stats.categories.keys())
        category_counts = list(stats.categories.values())
        
        axes[0, 0].bar(categories, category_counts)
        axes[0, 0].set_title('Category Distribution')
        axes[0, 0].set_xlabel('Categories')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Brands distribution (top 10)
        brands = list(stats.brands.keys())
        brand_counts = list(stats.brands.values())
        
        # Sort by count and take top 10
        brand_data = sorted(zip(brands, brand_counts), key=lambda x: x[1], reverse=True)[:10]
        top_brands, top_brand_counts = zip(*brand_data)
        
        axes[0, 1].bar(top_brands, top_brand_counts)
        axes[0, 1].set_title('Top 10 Brands Distribution')
        axes[0, 1].set_xlabel('Brands')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Dataset split
        split_counts = [len(self.annotations['train']), 
                       len(self.annotations['val']), 
                       len(self.annotations['test'])]
        split_labels = ['Train', 'Validation', 'Test']
        
        axes[1, 0].pie(split_counts, labels=split_labels, autopct='%1.1f%%')
        axes[1, 0].set_title('Dataset Split')
        
        # Attributes distribution
        attributes = list(stats.attributes.keys())
        attribute_counts = list(stats.attributes.values())
        
        axes[1, 1].bar(attributes, attribute_counts)
        axes[1, 1].set_title('Attributes Distribution')
        axes[1, 1].set_xlabel('Attributes')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Dataset statistics visualization saved to {save_path}")
    
    def export_dataset(self, export_path: str, format: str = 'coco'):
        """Export dataset in different formats"""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(exist_ok=True)
            
            if format == 'coco':
                self._export_coco_format(export_dir)
            elif format == 'yolo':
                self._export_yolo_format(export_dir)
            elif format == 'fashionpedia':
                self._export_fashionpedia_format(export_dir)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Dataset exported to {export_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return False
    
    def _export_coco_format(self, export_dir: Path):
        """Export dataset in COCO format"""
        # Create COCO annotation structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [],
            "info": {
                "description": "Fashion Recognition Dataset",
                "version": "1.0",
                "year": 2024
            }
        }
        
        # Add categories
        for i, category in enumerate(self.categories):
            coco_data["categories"].append({
                "id": i + 1,
                "name": category,
                "supercategory": "clothing"
            })
        
        # Add brands as supercategories
        for i, brand in enumerate(self.brands):
            coco_data["categories"].append({
                "id": i + len(self.categories) + 1,
                "name": brand,
                "supercategory": "brand"
            })
        
        # Process images and annotations
        image_id = 1
        annotation_id = 1
        
        for split, annotations in self.annotations.items():
            split_dir = export_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for ann in annotations:
                # Copy image to export directory
                src_path = Path(ann.image_path)
                if src_path.exists():
                    dst_path = split_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                    
                    # Add image info
                    with Image.open(src_path) as img:
                        width, height = img.size
                    
                    coco_data["images"].append({
                        "id": image_id,
                        "file_name": src_path.name,
                        "width": width,
                        "height": height,
                        "date_captured": "",
                        "license": 0
                    })
                    
                    # Add annotation
                    category_id = self.categories.index(ann.category) + 1
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": list(ann.bbox) if ann.bbox else [0, 0, width, height],
                        "area": (ann.bbox[2] * ann.bbox[3]) if ann.bbox else (width * height),
                        "iscrowd": 0,
                        "attributes": ann.attributes
                    })
                    
                    image_id += 1
                    annotation_id += 1
        
        # Save COCO annotation file
        with open(export_dir / "annotations.json", 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def _export_yolo_format(self, export_dir: Path):
        """Export dataset in YOLO format"""
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            split_dir = export_dir / split
            split_dir.mkdir(exist_ok=True)
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            for ann in self.annotations[split]:
                # Copy image
                src_path = Path(ann.image_path)
                if src_path.exists():
                    dst_path = images_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                    
                    # Create YOLO label file
                    label_file = labels_dir / (src_path.stem + '.txt')
                    
                    with Image.open(src_path) as img:
                        img_width, img_height = img.size
                    
                    with open(label_file, 'w') as f:
                        if ann.bbox:
                            # Convert bbox to YOLO format (normalized)
                            x, y, w, h = ann.bbox
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            width = w / img_width
                            height = h / img_height
                            
                            # Get category ID
                            category_id = self.categories.index(ann.category)
                            
                            f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
    
    def _export_fashionpedia_format(self, export_dir: Path):
        """Export dataset in Fashionpedia format"""
        # Similar to COCO but with Fashionpedia-specific structure
        fashionpedia_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "attributes": [],
            "licenses": [],
            "info": {
                "description": "Fashion Recognition Dataset - Fashionpedia Format",
                "version": "1.0",
                "year": 2024
            }
        }
        
        # Add categories (similar to COCO)
        for i, category in enumerate(self.categories):
            fashionpedia_data["categories"].append({
                "id": i + 1,
                "name": category,
                "supercategory": "clothing"
            })
        
        # Add attributes
        for i, attribute in enumerate(self.attributes):
            fashionpedia_data["attributes"].append({
                "id": i + 1,
                "name": attribute,
                "supercategory": "style"
            })
        
        # Process annotations (similar to COCO but with attributes)
        # Implementation similar to _export_coco_format but with attribute handling
        
        # Save Fashionpedia annotation file
        with open(export_dir / "fashionpedia_annotations.json", 'w') as f:
            json.dump(fashionpedia_data, f, indent=2)

def create_sample_dataset():
    """Create a sample dataset for testing"""
    manager = FashionDatasetManager("sample_fashion_dataset")
    
    # Add some sample annotations
    sample_annotations = [
        ("sample_shirt_1.jpg", "shirt", "Nike", ["red", "cotton", "casual"]),
        ("sample_pants_1.jpg", "pants", "Levi's", ["blue", "denim", "casual"]),
        ("sample_dress_1.jpg", "dress", "Chanel", ["black", "silk", "formal"]),
        ("sample_shoes_1.jpg", "shoes", "Adidas", ["white", "sporty"]),
        ("sample_jacket_1.jpg", "jacket", "North Face", ["green", "cotton", "casual"])
    ]
    
    for image_path, category, brand, attributes in sample_annotations:
        manager.add_image(image_path, category, brand, attributes, 'train')
    
    # Generate and save statistics
    stats = manager.get_dataset_stats()
    print(f"Dataset Statistics:")
    print(f"Total Images: {stats.total_images}")
    print(f"Categories: {stats.categories}")
    print(f"Brands: {stats.brands}")
    print(f"Attributes: {stats.attributes}")
    
    # Save annotations
    manager.save_annotations()
    
    return manager

def main():
    """Main function for dataset management"""
    # Create sample dataset
    manager = create_sample_dataset()
    
    # Visualize statistics
    manager.visualize_dataset_stats()
    
    # Export in different formats
    manager.export_dataset("exported_dataset", "coco")
    manager.export_dataset("exported_dataset_yolo", "yolo")

if __name__ == "__main__":
    main()
