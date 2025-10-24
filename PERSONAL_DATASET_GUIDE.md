# Adding Your Personal Clothing Dataset

This guide explains how to add your own clothing photos to the fashion recognition system.

## Quick Start

1. **Take Photos**: Capture clear photos of your clothes with good lighting
2. **Run the Script**: Use the provided `add_personal_dataset.py` script
3. **Train Model**: Use your new dataset to train the recognition model

## Step-by-Step Instructions

### 1. Prepare Your Images

- Take photos of your clothes with consistent lighting
- Use a plain background (white or neutral works best)
- Ensure the clothing item is clearly visible
- Save as JPG format
- Recommended size: 224x224 pixels or larger

### 2. Modify the Script

Edit `add_personal_dataset.py` and update the `clothing_items` list:

```python
clothing_items = [
    ("path/to/your/hoodie.jpg", "hoodie", "Nike", ["black", "cotton", "casual"]),
    ("path/to/your/jeans.jpg", "pants", "Levi's", ["blue", "denim", "casual"]),
    ("path/to/your/sweatpants.jpg", "sweatpants", "Adidas", ["gray", "cotton", "casual"]),
    ("path/to/your/dress.jpg", "dress", "Zara", ["red", "cotton", "formal"]),
    # Add more items...
]
```

### 3. Run the Script

```bash
python add_personal_dataset.py
```

### 4. Check Results

The script will:
- Create a new dataset folder: `my_personal_clothing_dataset/`
- Split your data into train/validation/test sets
- Generate statistics visualization
- Export the dataset in COCO format for training

## Supported Categories

- **Clothing Types**: shirt, pants, dress, jacket, shoes, hat, bag, watch, belt, tie, socks, underwear, swimwear, coat, sweater, hoodie, sweatpants
- **Brands**: Nike, Adidas, Gucci, Louis Vuitton, Chanel, Prada, Zara, H&M, Uniqlo, Gap, Levi's, and many more
- **Attributes**: 
  - Colors: red, blue, green, black, white, yellow, pink, purple
  - Materials: cotton, denim, leather, silk, wool
  - Patterns: striped, polka_dot, solid, patterned
  - Style: casual, formal, sporty

## Batch Processing

For multiple photos of the same type of clothing, use the folder method:

```python
# Add all images from a folder
add_images_from_folder(
    folder_path="path/to/your/shirts",
    category="shirt",
    brand="Your Brand",
    attributes=["blue", "cotton", "casual"]
)
```

## Training with Your Dataset

After adding your dataset, train the model:

```python
from model_training import FashionTrainer, FashionDataset
import torchvision.transforms as transforms

# Load your dataset
train_dataset = FashionDataset("my_personal_exported/train")
val_dataset = FashionDataset("my_personal_exported/val")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
trainer = FashionTrainer(model, device)
trainer.train(train_loader, val_loader, num_epochs=50)
```

## Tips for Better Results

1. **Consistent Lighting**: Use the same lighting setup for all photos
2. **Clear Backgrounds**: Plain backgrounds help the model focus on clothing
3. **Multiple Angles**: Take photos from different angles for better recognition
4. **Good Quality**: Use high-resolution images when possible
5. **Balanced Dataset**: Try to have similar numbers of each clothing type

## Troubleshooting

- **Image not found**: Check that the file path is correct
- **Unknown category/brand**: Use the supported categories and brands listed above
- **Low accuracy**: Ensure good image quality and consistent annotations
- **Memory issues**: Reduce image size or batch size during training

## File Structure

After running the script, you'll have:

```
my_personal_clothing_dataset/
├── train/                    # Training images (70%)
├── val/                     # Validation images (20%)
├── test/                    # Test images (10%)
├── annotations/             # JSON annotation files
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
└── my_personal_dataset_stats.png  # Statistics visualization

my_personal_exported/        # COCO format export
├── train/
├── val/
├── test/
└── annotations.json
```
