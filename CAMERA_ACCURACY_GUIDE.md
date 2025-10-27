# Improving Camera Accuracy for Personal Clothing Recognition

This guide explains how to train your model to achieve better camera accuracy for recognizing your personal clothing items.

## Quick Start

1. **Add your dataset**: Run `python add_personal_dataset.py`
2. **Train the model**: Run `python train_personal_model.py`
3. **Test accuracy**: Use the trained model with your camera

## Step-by-Step Training Process

### 1. Prepare Your Dataset for Maximum Accuracy

#### **Photo Quality Tips:**
- **Lighting**: Use bright, even lighting (natural light works best)
- **Background**: Use plain, contrasting backgrounds (white/black)
- **Angles**: Take photos from multiple angles (front, back, side)
- **Distance**: Fill the frame with the clothing item
- **Resolution**: Use high-resolution photos (at least 224x224 pixels)

#### **Dataset Size Recommendations:**
- **Minimum**: 10-15 photos per clothing category
- **Optimal**: 30-50 photos per category
- **Multiple items**: Include 3-5 different items per category

### 2. Run the Training Script

```bash
# First, add your personal dataset
python add_personal_dataset.py

# Then train the model
python train_personal_model.py
```

### 3. Monitor Training Progress

The script will show:
- Training and validation accuracy
- Loss curves
- Best model checkpoints
- Early stopping when accuracy plateaus

## Advanced Accuracy Improvements

### 1. **Data Augmentation for Real-World Conditions**

The training script includes enhanced augmentation to simulate real camera conditions:

- **Rotation**: Simulates different camera angles
- **Brightness/Contrast**: Handles varying lighting conditions
- **Color Jitter**: Accounts for different camera color profiles
- **Translation**: Simulates slight camera movement

### 2. **Class Balancing**

If you have imbalanced data (e.g., more shirts than dresses):
- The script automatically uses weighted sampling
- Prioritizes underrepresented classes during training
- Prevents the model from being biased toward common items

### 3. **Model Architecture Optimization**

The script uses:
- **ResNet50 backbone**: More powerful than basic CNN
- **Multi-task learning**: Simultaneously learns categories, brands, and attributes
- **Transfer learning**: Leverages pre-trained weights

## Camera-Specific Optimizations

### 1. **Lighting Conditions**

Train with photos taken in similar lighting to your camera usage:
- **Indoor lighting**: If you'll use it indoors, train with indoor photos
- **Outdoor lighting**: If you'll use it outdoors, include outdoor photos
- **Mixed lighting**: Include both for maximum robustness

### 2. **Camera Angles**

Include photos from angles you'll actually use:
- **Front view**: Most common for shirts, dresses
- **Side view**: Good for pants, shoes
- **Top view**: Useful for hats, bags
- **Multiple angles**: Helps the model generalize

### 3. **Background Variations**

Train with different backgrounds to improve real-world accuracy:
- **Plain backgrounds**: White, black, neutral colors
- **Textured backgrounds**: Wood, fabric, paper
- **Cluttered backgrounds**: Simulate real-world conditions

## Troubleshooting Low Accuracy

### **If accuracy is below 70%:**

1. **Check dataset quality**:
   - Are photos clear and well-lit?
   - Are backgrounds consistent?
   - Are clothing items clearly visible?

2. **Increase dataset size**:
   - Add more photos per category
   - Include more variety in poses/angles
   - Add photos in different lighting conditions

3. **Improve annotations**:
   - Double-check category labels
   - Ensure brand names are correct
   - Verify attribute labels

### **If accuracy is 70-85%:**

1. **Fine-tune hyperparameters**:
   - Increase learning rate slightly
   - Train for more epochs
   - Adjust batch size

2. **Add more diverse data**:
   - Include edge cases
   - Add photos in challenging conditions
   - Include similar-looking items

### **If accuracy is above 85%:**

1. **Test on real camera**:
   - Take photos with your actual camera
   - Test in different lighting conditions
   - Verify performance on new items

2. **Continuous improvement**:
   - Add new clothing items as you get them
   - Retrain periodically with new data
   - Monitor performance over time

## Real-Time Camera Testing

After training, test your model with real camera input:

```python
import cv2
import torch
from train_personal_model import load_trained_model

# Load your trained model
model = load_trained_model("best_personal_model_epoch_X.pth")
model.eval()

# Test with camera
cap = cv2.VideoCapture(0)  # Use your camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = transform(frame_pil).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        category_logits, brand_logits, _ = model(frame_tensor)
        category_pred = torch.argmax(category_logits, 1).item()
        brand_pred = torch.argmax(brand_logits, 1).item()
    
    # Display results
    cv2.putText(frame, f"Category: {category_pred}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Brand: {brand_pred}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Clothing Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Monitoring

### **Key Metrics to Track:**

1. **Category Accuracy**: How well it identifies clothing types
2. **Brand Accuracy**: How well it identifies brands
3. **Confidence Scores**: How certain the model is about predictions
4. **False Positives**: Items incorrectly identified
5. **False Negatives**: Items missed by the model

### **Improvement Strategies:**

1. **Data Quality**: Focus on high-quality, diverse photos
2. **Regular Retraining**: Add new data and retrain periodically
3. **Error Analysis**: Identify patterns in misclassifications
4. **A/B Testing**: Compare different model versions

## Expected Results

With a well-prepared dataset and proper training:

- **Good accuracy**: 80-90% for categories, 70-80% for brands
- **Fast inference**: Real-time recognition on modern hardware
- **Robust performance**: Works in various lighting conditions
- **Personalized**: Recognizes your specific clothing items

## Next Steps

1. **Start with a small dataset** (10-20 items) to test the process
2. **Gradually add more items** as you take photos
3. **Monitor performance** and adjust as needed
4. **Expand to more categories** once you're satisfied with initial results

Remember: The more diverse and high-quality your training data, the better your camera accuracy will be!
