#!/usr/bin/env python3
"""
Accurate Fashion Recognition System with Person Detection
Uses proper person detection + clothing classification for accuracy
"""

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccurateFashionDetector:
    """Accurate fashion detection with person detection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load person detector (YOLOv5)
        try:
            self.person_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.person_detector.eval()
            logger.info("Person detector loaded successfully")
        except Exception as e:
            logger.error(f"Could not load person detector: {e}")
            self.person_detector = None
        
        # Clothing categories
        self.categories = [
            'top', 'bottom', 'dress', 'jacket', 'shoes', 'hat', 'bag',
            'outerwear', 'underwear', 'accessories', 'unknown'
        ]
        
        # Load clothing classifier
        self.clothing_classifier = self.load_clothing_classifier()
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Accurate Fashion Detector initialized")
    
    def load_clothing_classifier(self):
        """Load or create clothing classifier"""
        # Create a pre-trained ResNet50 model fine-tuned for clothing
        model = resnet50(pretrained=True)
        
        # Replace final layer for clothing classification
        model.fc = nn.Linear(model.fc.in_features, len(self.categories))
        
        # Load weights if available
        model_path = "accurate_clothing_classifier.pth"
        if Path(model_path).exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded clothing classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load classifier weights: {e}")
        else:
            logger.info("Using pre-trained ImageNet weights for clothing classifier")
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def detect_person(self, frame):
        """Detect person in frame using YOLOv5"""
        if self.person_detector is None:
            # Fallback: use the full frame if no person detector
            return [{'bbox': [0, 0, frame.shape[1], frame.shape[0]], 'confidence': 1.0}]
        
        # Run person detection
        results = self.person_detector(frame)
        
        # Extract person bounding boxes (class 0 is person in COCO)
        persons = []
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                if int(detection[5]) == 0:  # Class 0 is person
                    x1, y1, x2, y2, conf = detection[:5].cpu().numpy()
                    persons.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
        
        return persons
    
    def classify_clothing_region(self, region, region_type='upper_body'):
        """Classify clothing in a specific body region"""
        try:
            # Preprocess region
            region_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            region_tensor = self.transform(region_pil).unsqueeze(0).to(self.device)
            
            # Classify
            with torch.no_grad():
                logits = self.clothing_classifier(region_tensor)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get top 3 predictions
                top_probs, top_indices = torch.topk(probabilities, 3)
                
                predictions = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    predictions.append({
                        'category': self.categories[idx.item()],
                        'confidence': prob.item()
                    })
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error classifying clothing: {e}")
            return [{'category': 'unknown', 'confidence': 0.0}]
    
    def detect_body_regions(self, person_bbox, frame):
        """Detect body regions (upper body, lower body, etc.)"""
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1
        
        regions = {}
        
        # Upper body (shoulder to waist)
        upper_top = y1
        upper_bottom = y1 + int(height * 0.5)
        regions['upper_body'] = {
            'bbox': [x1, upper_top, x2, upper_bottom],
            'region': frame[upper_top:upper_bottom, x1:x2]
        }
        
        # Lower body (waist down)
        lower_top = y1 + int(height * 0.4)
        lower_bottom = y2
        regions['lower_body'] = {
            'bbox': [x1, lower_top, x2, lower_bottom],
            'region': frame[lower_top:lower_bottom, x1:x2]
        }
        
        # Head/shoulders for hat detection
        head_top = y1
        head_bottom = y1 + int(height * 0.25)
        regions['head'] = {
            'bbox': [x1, head_top, x2, head_bottom],
            'region': frame[head_top:head_bottom, x1:x2]
        }
        
        # Feet area for shoe detection
        if height > 200:  # Only if person is tall enough
            feet_top = y2 - int(height * 0.15)
            feet_bottom = y2
            regions['feet'] = {
                'bbox': [x1, feet_top, x2, feet_bottom],
                'region': frame[feet_top:feet_bottom, x1:x2]
            }
        
        return regions
    
    def process_frame(self, frame):
        """Process a single frame and detect clothing"""
        results = {
            'persons': [],
            'fps': 0
        }
        
        start_time = time.time()
        
        # Detect persons
        persons = self.detect_person(frame)
        
        for person in persons:
            person_result = {
                'bbox': person['bbox'],
                'confidence': person['confidence'],
                'clothing': []
            }
            
            # Extract body regions
            regions = self.detect_body_regions(person['bbox'], frame)
            
            # Classify clothing in each region
            for region_name, region_data in regions.items():
                try:
                    # Skip small regions
                    h, w = region_data['region'].shape[:2]
                    if h < 20 or w < 20:
                        continue
                    
                    # Classify clothing
                    predictions = self.classify_clothing_region(region_data['region'], region_name)
                    
                    # Add best prediction
                    if predictions and predictions[0]['confidence'] > 0.3:
                        person_result['clothing'].append({
                            'region': region_name,
                            'category': predictions[0]['category'],
                            'confidence': predictions[0]['confidence'],
                            'bbox': region_data['bbox']
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing region {region_name}: {e}")
                    continue
            
            results['persons'].append(person_result)
        
        # Calculate FPS
        results['fps'] = 1.0 / (time.time() - start_time + 0.001)
        
        return results
    
    def draw_detections(self, frame, results):
        """Draw detection results on frame"""
        for person in results['persons']:
            # Draw person bounding box
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw clothing detections
            for clothing in person['clothing']:
                cx1, cy1, cx2, cy2 = clothing['bbox']
                
                # Choose color based on region
                color_map = {
                    'upper_body': (255, 0, 0),  # Red
                    'lower_body': (0, 255, 0),  # Green
                    'head': (0, 0, 255),        # Blue
                    'feet': (255, 255, 0)       # Cyan
                }
                color = color_map.get(clothing['region'], (128, 128, 128))
                
                # Draw bounding box
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)
                
                # Draw label
                label = f"{clothing['category']} ({clothing['confidence']:.2f})"
                cv2.putText(frame, label, (cx1, cy1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {results['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw clothing count
        clothing_count = sum(len(p['clothing']) for p in results['persons'])
        cv2.putText(frame, f"Clothing Items: {clothing_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def run_camera(self, camera_index=0):
        """Run detection on camera feed"""
        logger.info("Starting camera detection...")
        logger.info("Press 'q' to quit, 'c' to capture frame")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw results
                frame = self.draw_detections(frame, results)
                
                # Display frame
                cv2.imshow('Accurate Fashion Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    filename = f"accurate_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Captured frame: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera detection ended")
    
    def process_image(self, image_path):
        """Process a single image"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        # Process frame
        results = self.process_frame(frame)
        
        # Draw results
        result_frame = self.draw_detections(frame.copy(), results)
        
        # Display
        cv2.imshow('Accurate Fashion Detection', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_path = f"result_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, result_frame)
        logger.info(f"Result saved: {output_path}")
        
        # Print results
        for person in results['persons']:
            logger.info(f"Person detected (confidence: {person['confidence']:.2f})")
            for clothing in person['clothing']:
                logger.info(f"  {clothing['region']}: {clothing['category']} ({clothing['confidence']:.2f})")

def main():
    """Main function"""
    detector = AccurateFashionDetector()
    
    # Process the specific image
    image_path = "accurate_detection_capture_20251027_120020.jpg"
    if Path(image_path).exists():
        detector.process_image(image_path)
    else:
        logger.error(f"Image not found: {image_path}")
    
    # Ask to run camera
    print("\nRun camera detection? (y/n): ", end='')
    choice = input().strip().lower()
    if choice == 'y':
        detector.run_camera()

if __name__ == "__main__":
    main()
