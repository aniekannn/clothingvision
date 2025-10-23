Use model_training.py to retrain with more diverse or higher-quality images. See below for how to add your own dataset.

Adding Your Own Clothing Dataset
Use the FashionDatasetManager class from dataset_manager.py:

1. Initialize your dataset
python
from dataset_manager import FashionDatasetManager
manager = FashionDatasetManager("my_fashion_dataset")
2. Add images with annotations
python
manager.add_image("image1.jpg", "hoodie", "Adidas", ["black", "cotton", "sporty"])
manager.add_image("image2.jpg", "jeans", "Levi's", ["blue", "denim", "casual"])
Each image should be labeled with:

Clothing type

Brand

Attributes (color, material, style)

3. Split the dataset
python
manager.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
4. Export the dataset
python
manager.export_dataset("exported", "coco")
This prepares it for training in COCO format.

5. Train your model
Use FashionTrainer from model_training.py:

python
from model_training import FashionTrainer, FashionDataset

train_dataset = FashionDataset("data/train", transform=train_transform)
val_dataset = FashionDataset("data/val", transform=val_transform)

trainer = FashionTrainer(model, device)
trainer.train(train_loader, val_loader, num_epochs=50)