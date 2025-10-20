# Fashion Recognition System

A comprehensive real-time fashion recognition system that can identify clothing items and their brands using computer vision and deep learning. Built with PyTorch and OpenCV, this system provides high-accuracy detection and classification of fashion items from camera input.

CHECK USAGE_GUIDE FOR INFO ON HOW TO USE

---
Installation:

1. Clone the Repository
- git clone https://github.com/aniekannn/clothingvision.git
- cd clothingvision

2. Create Virtual Environment
- python -m venv fashion_env
- source fashion_env/bin/activate  (On macOS/Linux)

- fashion_env\\Scripts\\activate  (On Windows)

3. Install Dependencies
- pip install -r requirements_simple.txt

---
To Start:

1. Activate Virtual Environment
- source fashion_env/bin/activate  (On macOS/Linux)


2. Run the Accurate Fashion Detection Demo
Runs the main accurate detection demo (only detects clothing on your body)
- python accurate_camera_demo.py

---
Available Commands:

1. Run accurate fashion detection
- python accurate_camera_demo.py

2. Run individual system components
- python fashion_recognition_system.py
- python model_training.py
- python fashion_ui.py

3. Manage datasets
- python dataset_manager.py

---
Key Modules
- `accurate_camera_demo.py`: Main demo - Accurate fashion detection that only detects clothing on your body
- `accurate_detection.py`: Core accurate detection algorithm with person detection and clothing region mapping
- `fashion_recognition_system.py`: Main system with camera processing and detection
- `brand_recognition.py`: Advanced brand recognition with logo detection
- `fashion_ui.py`: User interface with real-time preview
- `model_training.py`: Model training utilities and evaluation
- `dataset_manager.py`: Dataset management and preprocessing
- `performance_optimizer.py`: Performance optimization and monitoring

---
To Create Dataset
- from dataset_manager import FashionDatasetManager

- Initialize dataset manager
manager = FashionDatasetManager("my_fashion_dataset")

- Add images with annotations
manager.add_image("image.jpg", "shirt", "Nike", ["red", "cotton", "casual"])

- Split dataset
manager.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

- Export in different formats
manager.export_dataset("exported", "coco")

---
Training

- Custom Model Training
from model_training import FashionTrainer, FashionDataset

- Create datasets
train_dataset = FashionDataset("data/train", transform=train_transform)
val_dataset = FashionDataset("data/val", transform=val_transform)

- Initialize trainer
trainer = FashionTrainer(model, device)

- Train model
trainer.train(train_loader, val_loader, num_epochs=50)