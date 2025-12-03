#!/usr/bin/env python3
"""
Simple but Accurate Clothing Detection Demo
Focuses on smooth movement and accurate detection
"""

import cv2
import numpy as np
import time
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleClothingDetector:
    """Simple detector with smooth tracking and accurate color detection"""
    
    def __init__(self):
        # Smoothing
        self.bbox_queue = deque(maxlen=7)  # Smooth over 7 frames
        
        # Initialize with center region
        self.current_bbox = None
        
        logger.info("Simple Clothing Detector initialized")
    
    def detect_dominant_color(self, region):
        """Detect the dominant color in a region"""
        if region is None or region.size == 0:
            return "unknown"
        
        # Resize for faster processing
        if region.shape[0] > 100:
            region = cv2.resize(region, (100, 100))
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Get average values
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)
        
        # Color classification based on HSV
        if avg_s < 30:  # Low saturation = grayscale
            if avg_v > 200:
                return "white"
            elif avg_v < 50:
                return "black"
            else:
                return "grey"
        
        # High saturation = actual color
        if 0 <= avg_h < 10 or 170 <= avg_h <= 180:
            return "red"
        elif 10 <= avg_h < 25:
            return "orange"
        elif 25 <= avg_h < 40:
            return "yellow"
        elif 40 <= avg_h < 80:
            return "green"
        elif 80 <= avg_h < 130:
            return "blue"
        elif 130 <= avg_h < 150:
            return "purple"
        elif 150 <= avg_h < 170:
            return "pink"
        else:
            return "brown"
    
    def get_person_bbox(self, frame):
        """Get smoothed bounding box for person"""
        height, width = frame.shape[:2]
        
        # Detect person using simple motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Use adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 20000:
                x, y, w, h = cv2.boundingRect(largest)
                
                # Add padding
                padding = 40
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                bbox = (x, y, w, h)
            else:
                # Fallback to center
                bbox = (width // 4, 0, width // 2, height)
        else:
            # Fallback to center
            bbox = (width // 4, 0, width // 2, height)
        
        # Add to queue for smoothing
        self.bbox_queue.append(bbox)
        
        # Calculate exponentially weighted average
        if len(self.bbox_queue) > 0:
            weights = np.linspace(0.5, 2.0, len(self.bbox_queue))
            weights = weights / weights.sum()
            
            avg_x = sum(b[0] * w for b, w in zip(self.bbox_queue, weights))
            avg_y = sum(b[1] * w for b, w in zip(self.bbox_queue, weights))
            avg_w = sum(b[2] * w for b, w in zip(self.bbox_queue, weights))
            avg_h = sum(b[3] * w for b, w in zip(self.bbox_queue, weights))
            
            self.current_bbox = (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
        else:
            self.current_bbox = bbox
        
        return self.current_bbox
    
    def detect_clothing(self, frame):
        """Detect clothing items on person"""
        # Get person bbox
        px, py, pw, ph = self.get_person_bbox(frame)
        
        detections = []
        
        # Define clothing regions (relative to person bbox)
        regions = {
            'shirt': (0.15, 0.60),   # Top 15% to 60% of person height
            'pants': (0.55, 0.90),    # 55% to 90% of person height
            'shoes': (0.85, 1.0)      # Bottom 85% to 100% of person height
        }
        
        for clothing_type, (y_start_ratio, y_end_ratio) in regions.items():
            # Calculate region coordinates
            ry1 = py + int(ph * y_start_ratio)
            ry2 = py + int(ph * y_end_ratio)
            rx1 = px + int(pw * 0.1)
            rx2 = px + int(pw * 0.9)
            
            # Ensure valid coordinates
            ry1 = max(0, ry1)
            ry2 = min(frame.shape[0], ry2)
            rx1 = max(0, rx1)
            rx2 = min(frame.shape[1], rx2)
            
            if ry2 > ry1 and rx2 > rx1:
                region = frame[ry1:ry2, rx1:rx2]
                
                # Detect color
                color = self.detect_dominant_color(region)
                
                detections.append({
                    'category': clothing_type,
                    'color': color,
                    'bbox': (rx1, ry1, rx2 - rx1, ry2 - ry1),
                    'confidence': 0.85
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        result = frame.copy()
        
        # Color map for boxes
        color_map = {
            'shirt': (0, 255, 0),   # Green
            'pants': (255, 0, 0),   # Blue
            'shoes': (0, 0, 255)    # Red
        }
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            category = detection['category']
            color_name = detection['color']
            
            box_color = color_map.get(category, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 3)
            
            # Draw label
            label = f"{category} ({color_name})"
            
            # Label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, (x, y - text_h - 10), (x + text_w, y), box_color, -1)
            
            # Label text
            cv2.putText(result, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result

def main():
    """Main function"""
    logger.info("Starting Simple Accurate Clothing Detection Demo")
    logger.info("Press 'q' to quit, 'c' to capture")
    
    # Initialize detector
    detector = SimpleClothingDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        logger.error("Run: tccutil reset Camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect clothing
            detections = detector.detect_clothing(frame)
            
            # Draw results
            result = detector.draw_detections(frame, detections)
            
            # Add FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result, f"Items: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Simple Accurate Detection', result)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                filename = f"simple_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, result)
                logger.info(f"Captured: {filename}")
    
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")

if __name__ == "__main__":
    main()


