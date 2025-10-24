#!/usr/bin/env python3
"""
Script to add your personal clothing dataset to the fashion recognition system.
This script demonstrates how to add your own clothes photos with proper annotations.
"""

import os
from pathlib import Path
from dataset_manager import FashionDatasetManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_personal_clothing_dataset():
    """
    Add your personal clothing photos to the dataset.
    Modify the clothing_items list below with your actual photos and details.
    """
    
    # Initialize dataset manager
    manager = FashionDatasetManager("my_personal_clothing_dataset")
    
    # Define your clothing items
    # Format: (image_path, category, brand, attributes)
    clothing_items = [
        # Example items - replace with your actual photos
        ("my_hoodie_1.jpg", "shirt", "Realtree", ["green", "camo", "casual"]),
        ("my_hoodie_1.jpg", "shirt", "Oldnavy", ["White", "Blue", "cotton"]),

        ("my_hoodie_1.jpg", "hoodie", "Yeezy", ["Black", "casual"]),
        ("my_hoodie_1.jpg", "hoodie", "Comfrt", ["Purple", "casual"]),
        ("my_hoodie_1.jpg", "hoodie", "Essentials", ["Tan", "White", "casual"]),
        ("my_hoodie_1.jpg", "hoodie", "Choate", ["Blue", "White", "Navy"]),

        ("my_sweatpants_1.jpg", "sweatpants", "Nike", ["blue", "vintage", "casual"]),
        ("my_sweatpants_1.jpg", "sweatpants", "Uniqlo", ["Gray", "Baggy", "casual"]),
        ("my_sweatpants_1.jpg", "sweatpants", "Choate", ["Gray", "casual"]),

        ("my_dress_1.jpg", "shoes", "Golden Goose", ["White", "Gray", "Sparkly"]),
        ("my_sneakers_1.jpg", "shoes", "Adidas", ["Gray", "sporty"]),
        ("my_sneakers_1.jpg", "shoes", "Adidas", ["Black", "sporty", "Sambas"]),

        ("my_jacket_1.jpg", "hat", "Supreme", ["gray", "Black", "red", "casual"]),
        ("my_jacket_1.jpg", "hat", "Stussy", ["Black", "Sporty"]),
        

        
        ("my_shirt_1.jpg", "shirt", "Uniqlo", ["white", "cotton", "casual"]),
        ("my_pants_2.jpg", "pants", "H&M", ["black", "cotton", "formal"]),
        ("my_hat_1.jpg", "hat", "Nike", ["red", "cotton", "sporty"]),
        # Add more items as needed...
    ]
    
    # Add each item to the dataset
    added_count = 0
    for image_path, category, brand, attributes in clothing_items:
        # Check if image file exists
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path} - skipping")
            continue
            
        # Add to training set initially
        success = manager.add_image(
            image_path=image_path,
            category=category,
            brand=brand,
            attributes=attributes,
            split="train"  # Will be redistributed later
        )
        
        if success:
            added_count += 1
            logger.info(f"Added: {image_path} -> {category} ({brand})")
        else:
            logger.error(f"Failed to add: {image_path}")
    
    logger.info(f"Successfully added {added_count} clothing items")
    
    # Split dataset into train/val/test
    logger.info("Splitting dataset...")
    success = manager.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    if success:
        logger.info("Dataset split completed successfully")
    else:
        logger.error("Failed to split dataset")
        return False
    
    # Save annotations
    manager.save_annotations()
    logger.info("Annotations saved")
    
    # Get and display statistics
    stats = manager.get_dataset_stats()
    logger.info("Dataset Statistics:")
    logger.info(f"  Total Images: {stats.total_images}")
    logger.info(f"  Categories: {list(stats.categories.keys())}")
    logger.info(f"  Brands: {list(stats.brands.keys())}")
    logger.info(f"  Average Image Size: {stats.avg_image_size}")
    logger.info(f"  Total File Size: {stats.file_size_mb:.2f} MB")
    
    # Visualize dataset statistics
    manager.visualize_dataset_stats("my_personal_dataset_stats.png")
    logger.info("Dataset statistics visualization saved as 'my_personal_dataset_stats.png'")
    
    # Export dataset for training
    manager.export_dataset("my_personal_exported", "coco")
    logger.info("Dataset exported in COCO format to 'my_personal_exported'")
    
    return True

def add_images_from_folder(folder_path: str, category: str, brand: str, attributes: list):
    """
    Add all images from a folder with the same category, brand, and attributes.
    Useful when you have multiple photos of the same type of clothing.
    
    Args:
        folder_path: Path to folder containing images
        category: Clothing category (e.g., "shirt", "pants")
        brand: Brand name
        attributes: List of attributes (e.g., ["red", "cotton", "casual"])
    """
    manager = FashionDatasetManager("my_personal_clothing_dataset")
    
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return False
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.error(f"No image files found in {folder_path}")
        return False
    
    logger.info(f"Found {len(image_files)} images in {folder_path}")
    
    # Add each image
    added_count = 0
    for image_file in image_files:
        success = manager.add_image(
            image_path=str(image_file),
            category=category,
            brand=brand,
            attributes=attributes,
            split="train"
        )
        
        if success:
            added_count += 1
            logger.info(f"Added: {image_file.name}")
        else:
            logger.error(f"Failed to add: {image_file.name}")
    
    logger.info(f"Successfully added {added_count} images from folder")
    return added_count > 0

def main():
    """Main function to run the dataset addition process"""
    print("Personal Clothing Dataset Addition Tool")
    print("=" * 50)
    
    # Option 1: Add individual items (modify the clothing_items list above)
    print("Adding individual clothing items...")
    success = add_personal_clothing_dataset()
    
    if success:
        print("\n✅ Dataset addition completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated statistics visualization")
        print("2. Use the exported dataset for training")
        print("3. Run model training with your new dataset")
    else:
        print("\n❌ Dataset addition failed. Check the logs for details.")
    
    # Option 2: Add images from folders (uncomment to use)
    # print("\nAdding images from folders...")
    # add_images_from_folder("path/to/your/shirts", "shirt", "Your Brand", ["blue", "cotton", "casual"])
    # add_images_from_folder("path/to/your/pants", "pants", "Your Brand", ["black", "denim", "casual"])

if __name__ == "__main__":
    main()
