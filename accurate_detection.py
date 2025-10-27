"""
Accurate Fashion Detection System
Only detects clothing items on the person's body, not background objects
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccurateFashionDetector:
    """Accurate fashion detector that only identifies clothing on the person"""
    
    def __init__(self):
        # Initialize detection parameters
        self.min_confidence = 0.4
        self.nms_threshold = 0.3
        
        # Person detection parameters
        self.person_min_area = 15000  # Minimum area for person detection
        self.person_max_area = 200000  # Maximum area for person detection
        
        # Clothing detection regions (relative to person bounding box)
        self.clothing_regions = {
            'shirt': {'y_ratio': (0.2, 0.6), 'x_ratio': (0.15, 0.85)},  # Upper torso
            'pants': {'y_ratio': (0.55, 0.9), 'x_ratio': (0.2, 0.8)},   # Lower torso to knees
            'shoes': {'y_ratio': (0.85, 1.0), 'x_ratio': (0.1, 0.9)},   # Feet area
        }
        
        # Color ranges for clothing detection (HSV)
        self.clothing_colors = {
            'shirt': {
                'grey': [(0, 0, 50), (180, 30, 150)],
                'white': [(0, 0, 150), (180, 30, 255)],
                'blue': [(100, 50, 50), (130, 255, 255)],
                'red': [(0, 50, 50), (10, 255, 255)],
                'green': [(40, 50, 50), (80, 255, 255)],
                'black': [(0, 0, 0), (180, 255, 50)],
            },
            'pants': {
                'blue': [(100, 50, 50), (130, 255, 255)],
                'black': [(0, 0, 0), (180, 255, 50)],
                'grey': [(0, 0, 50), (180, 30, 150)],
                'brown': [(10, 50, 50), (20, 255, 200)],
            },
            'shoes': {
                'black': [(0, 0, 0), (180, 255, 50)],
                'white': [(0, 0, 150), (180, 30, 255)],
                'brown': [(10, 50, 50), (20, 255, 200)],
                'blue': [(100, 50, 50), (130, 255, 255)],
            }
        }
        
        logger.info("Accurate Fashion Detector initialized")
    
    def detect_person_accurately(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect person using improved contour analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction-like effect
        # Use adaptive threshold to find person silhouette
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Use Otsu's thresholding for better segmentation
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.person_min_area < area < self.person_max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                # Person should be taller than wide (aspect ratio > 1.2)
                if aspect_ratio > 1.2:
                    valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
        
        # Get the largest valid contour
        largest_contour, _ = max(valid_contours, key=lambda x: x[1])
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def extract_clothing_region(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int], 
                              clothing_type: str) -> Optional[np.ndarray]:
        """Extract specific clothing region from person"""
        x, y, w, h = person_bbox
        region_info = self.clothing_regions.get(clothing_type)
        
        if not region_info:
            return None
        
        # Calculate region coordinates
        y_start = y + int(h * region_info['y_ratio'][0])
        y_end = y + int(h * region_info['y_ratio'][1])
        x_start = x + int(w * region_info['x_ratio'][0])
        x_end = x + int(w * region_info['x_ratio'][1])
        
        # Ensure valid coordinates
        y_start = max(0, y_start)
        y_end = min(frame.shape[0], y_end)
        x_start = max(0, x_start)
        x_end = min(frame.shape[1], x_end)
        
        if y_end <= y_start or x_end <= x_start:
            return None
        
        return frame[y_start:y_end, x_start:x_end]
    
    def analyze_clothing_color(self, region: np.ndarray, clothing_type: str) -> Optional[Dict]:
        """Analyze clothing region for color and confidence"""
        if region is None or region.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Get color ranges for this clothing type
        color_ranges = self.clothing_colors.get(clothing_type, {})
        
        best_color = None
        best_confidence = 0
        
        # Test each color range
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Calculate confidence based on mask coverage
            total_pixels = region.shape[0] * region.shape[1]
            colored_pixels = cv2.countNonZero(mask)
            confidence = colored_pixels / total_pixels
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_color = color_name
        
        # Only return detection if confidence is above threshold
        if best_confidence > self.min_confidence:
            return {
                'color': best_color,
                'confidence': best_confidence
            }
        
        return None
    
    def detect_clothing_on_person(self, frame: np.ndarray) -> List[Dict]:
        """Detect clothing items only on the person"""
        detections = []
        
        # First, detect the person accurately
        person_bbox = self.detect_person_accurately(frame)
        
        if person_bbox is None:
            return detections
        
        # Detect clothing items within the person region only
        for clothing_type in self.clothing_regions.keys():
            # Extract clothing region
            clothing_region = self.extract_clothing_region(frame, person_bbox, clothing_type)
            
            if clothing_region is not None:
                # Analyze color
                color_analysis = self.analyze_clothing_color(clothing_region, clothing_type)
                
                if color_analysis:
                    # Calculate bounding box in frame coordinates
                    x, y, w, h = person_bbox
                    region_info = self.clothing_regions[clothing_type]
                    
                    bbox_x = x + int(w * region_info['x_ratio'][0])
                    bbox_y = y + int(h * region_info['y_ratio'][0])
                    bbox_w = int(w * (region_info['x_ratio'][1] - region_info['x_ratio'][0]))
                    bbox_h = int(h * (region_info['y_ratio'][1] - region_info['y_ratio'][0]))
                    
                    detections.append({
                        'category': clothing_type,
                        'confidence': color_analysis['confidence'],
                        'color': color_analysis['color'],
                        'bbox': (bbox_x, bbox_y, bbox_w, bbox_h)
                    })
        
        return detections
    
    def draw_clothing_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw clothing detection results on frame"""
        result_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            category = detection['category']
            confidence = detection['confidence']
            color = detection['color']
            
            # Choose color for bounding box
            if category == 'shirt':
                box_color = (0, 255, 0)  # Green
            elif category == 'pants':
                box_color = (255, 0, 0)  # Blue
            elif category == 'shoes':
                box_color = (0, 0, 255)  # Red
            else:
                box_color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw label
            label = f"{category} ({color})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), box_color, -1)
            
            # Draw label text
            cv2.putText(result_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"{confidence:.2f}"
            cv2.putText(result_frame, conf_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        return result_frame
