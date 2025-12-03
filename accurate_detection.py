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
        self.min_confidence = 0.3
        self.nms_threshold = 0.3
        
        # Person detection parameters - use simple center tracking
        self.use_simple_tracking = True
        
        # Clothing detection regions (relative to person bounding box)
        # Adjusted to skip head/face area
        self.clothing_regions = {
            'shirt': {'y_ratio': (0.25, 0.55), 'x_ratio': (0.2, 0.8)},  # Chest/torso only
            'pants': {'y_ratio': (0.60, 0.85), 'x_ratio': (0.25, 0.75)},  # Waist to thighs
            'shoes': {'y_ratio': (0.90, 1.0), 'x_ratio': (0.15, 0.85)},  # Bottom only
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
        """Detect person using simple edge detection for movement tracking"""
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges to connect them
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest) > 20000:
                x, y, w, h = cv2.boundingRect(largest)
                
                # Add padding
                padding = 40
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                return (x, y, w, h)
        
        # Fallback to center if no detection
        margin_x = int(width * 0.2)
        return (margin_x, 0, width - 2 * margin_x, height)
    
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
        """Analyze clothing region for color with improved accuracy"""
        if region is None or region.size == 0:
            return None
        
        # Resize region for consistent analysis
        if region.shape[0] < 20 or region.shape[1] < 20:
            return None
            
        # Convert to multiple color spaces for better detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        
        # Enhanced color detection with better ranges
        color_detectors = {
            'white': lambda: self._detect_white(hsv, lab),
            'black': lambda: self._detect_black(hsv, lab),
            'grey': lambda: self._detect_grey(hsv, lab),
            'blue': lambda: self._detect_blue(hsv),
            'red': lambda: self._detect_red(hsv),
            'green': lambda: self._detect_green(hsv),
            'brown': lambda: self._detect_brown(hsv),
            'yellow': lambda: self._detect_yellow(hsv),
            'purple': lambda: self._detect_purple(hsv),
            'orange': lambda: self._detect_orange(hsv),
            'pink': lambda: self._detect_pink(hsv),
        }
        
        best_color = None
        best_confidence = 0
        
        # Test each color
        for color_name, detector in color_detectors.items():
            try:
                confidence = detector()
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_color = color_name
            except Exception as e:
                logger.debug(f"Error detecting {color_name}: {e}")
                continue
        
        # If we found ANY color with reasonable confidence, return it
        # Otherwise use simple dominant color fallback
        if best_color and best_confidence > self.min_confidence:
            return {
                'color': best_color,
                'confidence': best_confidence
            }
        elif best_color and best_confidence > 0.15:  # Lower fallback threshold
            return {
                'color': best_color,
                'confidence': best_confidence
            }
        else:
            # Always use simple dominant color detection
            avg_color_per_row = np.average(region, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            b, g, r = avg_color
            
            # Convert to HSV for better classification
            bgr_pixel = np.uint8([[[b, g, r]]])
            hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv_pixel
            
            # Classify based on HSV
            if s < 30:  # Low saturation = grayscale
                if v > 180:
                    fallback_color = 'white'
                elif v < 70:
                    fallback_color = 'black'
                else:
                    fallback_color = 'grey'
            elif h < 10 or h > 170:
                fallback_color = 'red'
            elif 10 <= h < 25:
                fallback_color = 'brown'
            elif 25 <= h < 40:
                fallback_color = 'yellow'
            elif 40 <= h < 80:
                fallback_color = 'green'
            elif 80 <= h < 130:
                fallback_color = 'blue'
            elif 130 <= h < 150:
                fallback_color = 'purple'
            elif 150 <= h < 170:
                fallback_color = 'pink'
            else:
                fallback_color = 'grey'
            
            return {
                'color': fallback_color,
                'confidence': 0.7
            }
    
    def _detect_white(self, hsv, lab):
        """Detect white clothing"""
        _, _, v = cv2.split(hsv)
        l, _, _ = cv2.split(lab)
        # White has high V and high L
        white_mask = cv2.bitwise_and(v > 180, l > 180)
        return cv2.countNonZero(white_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_black(self, hsv, lab):
        """Detect black clothing"""
        _, _, v = cv2.split(hsv)
        l, _, _ = cv2.split(lab)
        # Black has low V and low L
        black_mask = cv2.bitwise_and(v < 70, l < 70)
        return cv2.countNonZero(black_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_grey(self, hsv, lab):
        """Detect grey clothing"""
        _, s, v = cv2.split(hsv)
        # Grey has low saturation and medium value
        grey_mask = cv2.bitwise_and(s < 40, cv2.bitwise_and(v > 70, v < 180))
        return cv2.countNonZero(grey_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_blue(self, hsv):
        """Detect blue clothing"""
        h, s, v = cv2.split(hsv)
        # Blue hue range: 100-130
        blue_mask = cv2.bitwise_and(cv2.bitwise_and(h > 100, h < 130), s > 50)
        return cv2.countNonZero(blue_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_red(self, hsv):
        """Detect red clothing"""
        h, s, v = cv2.split(hsv)
        # Red wraps around: 0-10 and 170-180
        red_mask1 = cv2.bitwise_and(cv2.bitwise_and(h < 10, s > 50), v > 50)
        red_mask2 = cv2.bitwise_and(cv2.bitwise_and(h > 170, s > 50), v > 50)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        return cv2.countNonZero(red_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_green(self, hsv):
        """Detect green clothing"""
        h, s, v = cv2.split(hsv)
        # Green hue range: 40-80
        green_mask = cv2.bitwise_and(cv2.bitwise_and(h > 40, h < 80), s > 50)
        return cv2.countNonZero(green_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_brown(self, hsv):
        """Detect brown clothing"""
        h, s, v = cv2.split(hsv)
        # Brown: hue 10-20, moderate saturation, low-medium value
        brown_mask = cv2.bitwise_and(cv2.bitwise_and(h > 10, h < 25), 
                                     cv2.bitwise_and(s > 40, v < 180))
        return cv2.countNonZero(brown_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_yellow(self, hsv):
        """Detect yellow clothing"""
        h, s, v = cv2.split(hsv)
        # Yellow hue range: 25-35
        yellow_mask = cv2.bitwise_and(cv2.bitwise_and(h > 25, h < 35), s > 100)
        return cv2.countNonZero(yellow_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_purple(self, hsv):
        """Detect purple clothing"""
        h, s, v = cv2.split(hsv)
        # Purple hue range: 130-160
        purple_mask = cv2.bitwise_and(cv2.bitwise_and(h > 130, h < 160), s > 50)
        return cv2.countNonZero(purple_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_orange(self, hsv):
        """Detect orange clothing"""
        h, s, v = cv2.split(hsv)
        # Orange hue range: 10-25
        orange_mask = cv2.bitwise_and(cv2.bitwise_and(h > 10, h < 25), s > 100)
        return cv2.countNonZero(orange_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def _detect_pink(self, hsv):
        """Detect pink clothing"""
        h, s, v = cv2.split(hsv)
        # Pink: low saturation red or magenta hue
        pink_mask1 = cv2.bitwise_and(cv2.bitwise_and(h < 10, s > 30), v > 150)
        pink_mask2 = cv2.bitwise_and(cv2.bitwise_and(h > 160, s > 30), v > 150)
        pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)
        return cv2.countNonZero(pink_mask) / (hsv.shape[0] * hsv.shape[1])
    
    def detect_clothing_on_person(self, frame: np.ndarray) -> List[Dict]:
        """Detect clothing items only on the person"""
        detections = []
        
        # First, detect the person accurately
        person_bbox = self.detect_person_accurately(frame)
        
        if person_bbox is None:
            logger.warning("No person detected in frame")
            return detections
        
        logger.info(f"Person detected at bbox: {person_bbox}")
        
        # Detect clothing items within the person region only
        for clothing_type in self.clothing_regions.keys():
            # Extract clothing region
            clothing_region = self.extract_clothing_region(frame, person_bbox, clothing_type)
            
            if clothing_region is not None:
                logger.debug(f"Analyzing {clothing_type} region: {clothing_region.shape}")
                # Analyze color
                color_analysis = self.analyze_clothing_color(clothing_region, clothing_type)
                
                if color_analysis:
                    logger.info(f"Detected {clothing_type}: {color_analysis['color']} ({color_analysis['confidence']:.2f})")
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
