"""
Advanced Brand Recognition System
Specialized models for logo detection, brand identification, and pattern matching
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class LogoDetection:
    """Data class for logo detection results"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    brand_name: str
    logo_type: str  # 'text', 'symbol', 'combined'

@dataclass
class BrandSignature:
    """Data class for brand signature patterns"""
    name: str
    signature_type: str  # 'color', 'pattern', 'style', 'texture'
    confidence: float
    region: Tuple[int, int, int, int]

class LogoDetectorModel(nn.Module):
    """CNN model for logo detection and classification"""
    
    def __init__(self, num_brands: int = 50, num_logo_types: int = 3):
        super(LogoDetectorModel, self).__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Brand classification head
        self.brand_classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_brands)
        )
        
        # Logo type classification head
        self.logo_type_classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_logo_types)
        )
        
        # Object detection head (simplified)
        self.detection_head = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # x, y, w, h
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        
        brand_logits = self.brand_classifier(features_flat)
        logo_type_logits = self.logo_type_classifier(features_flat)
        detection_coords = torch.sigmoid(self.detection_head(features_flat))
        
        return brand_logits, logo_type_logits, detection_coords

class PatternRecognizer:
    """Pattern recognition for brand signatures"""
    
    def __init__(self):
        self.pattern_templates = self._load_pattern_templates()
        self.color_signatures = self._load_color_signatures()
    
    def _load_pattern_templates(self) -> Dict[str, np.ndarray]:
        """Load pattern templates for different brands"""
        patterns = {}
        
        # Simulate pattern templates (in reality, these would be loaded from files)
        brands = ['Gucci', 'Louis Vuitton', 'Burberry', 'Coach', 'Chanel']
        
        for brand in brands:
            # Create synthetic pattern templates
            if brand == 'Gucci':
                # Simulate Gucci's interlocking G pattern
                pattern = self._create_gucci_pattern()
            elif brand == 'Louis Vuitton':
                # Simulate LV monogram pattern
                pattern = self._create_lv_pattern()
            elif brand == 'Burberry':
                # Simulate Burberry plaid
                pattern = self._create_burberry_pattern()
            else:
                # Generic pattern
                pattern = np.random.rand(64, 64, 3)
            
            patterns[brand] = pattern
        
        return patterns
    
    def _load_color_signatures(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Load color signatures for different brands"""
        return {
            'Chanel': [(255, 255, 255), (0, 0, 0)],  # Black and white
            'Gucci': [(0, 128, 0), (255, 255, 255)],  # Green and white
            'Louis Vuitton': [(139, 69, 19), (255, 255, 255)],  # Brown and white
            'Tiffany': [(0, 255, 255), (255, 255, 255)],  # Tiffany blue and white
            'Hermès': [(255, 165, 0), (255, 255, 255)]  # Orange and white
        }
    
    def _create_gucci_pattern(self) -> np.ndarray:
        """Create a synthetic Gucci pattern"""
        pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        pattern.fill(255)  # White background
        
        # Add some synthetic pattern elements
        for i in range(0, 64, 16):
            for j in range(0, 64, 16):
                if (i + j) % 32 == 0:
                    pattern[i:i+8, j:j+8] = [0, 128, 0]  # Green
        
        return pattern
    
    def _create_lv_pattern(self) -> np.ndarray:
        """Create a synthetic LV pattern"""
        pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        pattern.fill(255)  # White background
        
        # Add some synthetic LV-like elements
        for i in range(0, 64, 12):
            for j in range(0, 64, 12):
                if (i + j) % 24 == 0:
                    pattern[i:i+6, j:j+6] = [139, 69, 19]  # Brown
        
        return pattern
    
    def _create_burberry_pattern(self) -> np.ndarray:
        """Create a synthetic Burberry plaid pattern"""
        pattern = np.zeros((64, 64, 3), dtype=np.uint8)
        pattern.fill(255)  # White background
        
        # Add plaid-like stripes
        for i in range(0, 64, 4):
            if i % 8 == 0:
                pattern[i:i+2, :] = [0, 0, 0]  # Black stripe
            if i % 8 == 4:
                pattern[:, i:i+2] = [0, 0, 0]  # Black stripe
        
        return pattern
    
    def detect_patterns(self, image: np.ndarray) -> List[BrandSignature]:
        """Detect brand patterns in the image"""
        signatures = []
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Template matching for patterns
        for brand, template in self.pattern_templates.items():
            # Resize template to match image scale
            h, w = image.shape[:2]
            template_resized = cv2.resize(template, (w//4, h//4))
            
            # Ensure both images have the same data type and number of channels
            if rgb_image.dtype != template_resized.dtype:
                template_resized = template_resized.astype(rgb_image.dtype)
            
            # Skip if template is larger than image
            if template_resized.shape[0] > h or template_resized.shape[1] > w:
                continue
                
            # Template matching
            result = cv2.matchTemplate(rgb_image, template_resized, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            locations = np.where(result >= 0.6)
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                region = (pt[0], pt[1], template_resized.shape[1], template_resized.shape[0])
                
                signatures.append(BrandSignature(
                    name=brand,
                    signature_type='pattern',
                    confidence=confidence,
                    region=region
                ))
        
        # Color signature detection
        color_signatures = self._detect_color_signatures(rgb_image)
        signatures.extend(color_signatures)
        
        return signatures
    
    def _detect_color_signatures(self, image: np.ndarray) -> List[BrandSignature]:
        """Detect brand-specific color signatures"""
        signatures = []
        
        # Get dominant colors
        dominant_colors = self._get_dominant_colors(image)
        
        # Compare with brand color signatures
        for brand, brand_colors in self.color_signatures.items():
            for brand_color in brand_colors:
                for dom_color, confidence in dominant_colors:
                    # Calculate color similarity
                    similarity = self._color_similarity(dom_color, brand_color)
                    
                    if similarity > 0.8:  # High similarity threshold
                        signatures.append(BrandSignature(
                            name=brand,
                            signature_type='color',
                            confidence=similarity,
                            region=(0, 0, image.shape[1], image.shape[0])
                        ))
        
        return signatures
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[Tuple[int, int, int], float]]:
        """Get dominant colors in the image using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Sample pixels for efficiency
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and counts
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Calculate proportions
        proportions = []
        for i in range(k):
            proportion = np.sum(labels == i) / len(labels)
            proportions.append(proportion)
        
        # Combine colors and proportions
        dominant_colors = list(zip([tuple(color) for color in colors], proportions))
        dominant_colors.sort(key=lambda x: x[1], reverse=True)  # Sort by proportion
        
        return dominant_colors
    
    def _color_similarity(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate similarity between two colors"""
        # Convert to numpy arrays
        c1 = np.array(color1)
        c2 = np.array(color2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(c1 - c2)
        
        # Convert to similarity (0-1 scale)
        max_distance = np.sqrt(3 * 255**2)  # Maximum possible distance
        similarity = 1 - (distance / max_distance)
        
        return similarity

class TextRecognizer:
    """Text recognition for brand names and logos"""
    
    def __init__(self):
        self.brand_keywords = self._load_brand_keywords()
    
    def _load_brand_keywords(self) -> Dict[str, List[str]]:
        """Load brand keywords for text recognition"""
        return {
            'Nike': ['nike', 'just do it', 'swoosh'],
            'Adidas': ['adidas', 'three stripes', 'impossible is nothing'],
            'Gucci': ['gucci', 'gg'],
            'Louis Vuitton': ['louis vuitton', 'lv', 'louis'],
            'Chanel': ['chanel', 'cc'],
            'Prada': ['prada'],
            'Versace': ['versace', 'medusa'],
            'Armani': ['armani', 'giorgio armani'],
            'Calvin Klein': ['calvin klein', 'ck'],
            'Tommy Hilfiger': ['tommy hilfiger', 'tommy'],
            'Ralph Lauren': ['ralph lauren', 'polo'],
            'Zara': ['zara'],
            'H&M': ['h&m', 'hm'],
            'Uniqlo': ['uniqlo'],
            'Gap': ['gap'],
            'Levi\'s': ['levi\'s', 'levis'],
            'Diesel': ['diesel'],
            'Guess': ['guess'],
            'Coach': ['coach'],
            'Michael Kors': ['michael kors', 'mk'],
            'Kate Spade': ['kate spade'],
            'Fendi': ['fendi'],
            'Balenciaga': ['balenciaga'],
            'Saint Laurent': ['saint laurent', 'ysl'],
            'Burberry': ['burberry'],
            'Hermès': ['hermes', 'hermès'],
            'Valentino': ['valentino'],
            'Dior': ['dior'],
            'Givenchy': ['givenchy'],
            'Bottega Veneta': ['bottega veneta']
        }
    
    def extract_text(self, image: np.ndarray) -> List[str]:
        """Extract text from image using OCR"""
        # For this implementation, we'll simulate text extraction
        # In practice, you would use Tesseract or similar OCR engine
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours that might contain text
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter by aspect ratio (text typically has certain aspect ratios)
            if 0.1 < aspect_ratio < 10 and w > 20 and h > 10:
                text_regions.append((x, y, w, h))
        
        # Simulate text extraction
        # In reality, you would run OCR on each text region
        extracted_text = []
        for region in text_regions:
            # Simulate OCR result
            simulated_text = self._simulate_ocr(image, region)
            if simulated_text:
                extracted_text.append(simulated_text)
        
        return extracted_text
    
    def _simulate_ocr(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """Simulate OCR for a text region"""
        # This is a placeholder - in reality, you would use Tesseract or similar
        x, y, w, h = region
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        
        # Simple heuristic based on image characteristics
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_roi)
        
        # Simulate text detection based on intensity
        if mean_intensity < 100:  # Dark text on light background
            # Return a random brand name for demonstration
            brands = list(self.brand_keywords.keys())
            return np.random.choice(brands)
        
        return ""
    
    def recognize_brands_from_text(self, extracted_text: List[str]) -> List[BrandSignature]:
        """Recognize brands from extracted text"""
        signatures = []
        
        for text in extracted_text:
            text_lower = text.lower()
            
            for brand, keywords in self.brand_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        # Calculate confidence based on text quality
                        confidence = len(keyword) / len(text) if text else 0.5
                        
                        signatures.append(BrandSignature(
                            name=brand,
                            signature_type='text',
                            confidence=confidence,
                            region=(0, 0, 0, 0)  # Text region coordinates
                        ))
        
        return signatures

class AdvancedBrandRecognizer:
    """Advanced brand recognition system combining multiple approaches"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Brand recognizer using device: {self.device}")
        
        # Initialize models
        self.logo_detector = LogoDetectorModel()
        self.pattern_recognizer = PatternRecognizer()
        self.text_recognizer = TextRecognizer()
        
        # Load model weights if available
        if model_path and os.path.exists(model_path):
            try:
                self.logo_detector.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Brand recognition model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}")
        
        self.logo_detector.to(self.device)
        self.logo_detector.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Brand names for classification
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
        
        self.logo_types = ['text', 'symbol', 'combined']
    
    def detect_logos(self, image: np.ndarray) -> List[LogoDetection]:
        """Detect and classify logos in the image"""
        detections = []
        
        # Find potential logo regions using contour detection
        logo_regions = self._find_logo_regions(image)
        
        for region in logo_regions:
            x, y, w, h = region
            
            # Extract region
            roi = image[y:y+h, x:x+w]
            
            # Skip very small regions
            if w < 20 or h < 20:
                continue
            
            # Preprocess for model input
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)
            roi_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                brand_logits, logo_type_logits, detection_coords = self.logo_detector(roi_tensor)
                
                # Apply softmax to get probabilities
                brand_probs = F.softmax(brand_logits, dim=1)
                logo_type_probs = F.softmax(logo_type_logits, dim=1)
                
                # Get top predictions
                brand_conf, brand_idx = torch.max(brand_probs, 1)
                logo_type_conf, logo_type_idx = torch.max(logo_type_probs, 1)
                
                # Filter by confidence threshold
                if brand_conf.item() > 0.3:  # Adjust threshold as needed
                    detections.append(LogoDetection(
                        bbox=region,
                        confidence=brand_conf.item(),
                        brand_name=self.brands[brand_idx.item()],
                        logo_type=self.logo_types[logo_type_idx.item()]
                    ))
        
        return detections
    
    def _find_logo_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find potential logo regions in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 50000:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Filter by aspect ratio (logos typically have reasonable aspect ratios)
                if 0.2 < aspect_ratio < 5.0:
                    regions.append((x, y, w, h))
        
        return regions
    
    def recognize_brands(self, image: np.ndarray) -> List[Dict]:
        """Comprehensive brand recognition combining all methods"""
        results = []
        
        # 1. Logo detection
        logo_detections = self.detect_logos(image)
        for detection in logo_detections:
            results.append({
                'method': 'logo_detection',
                'brand': detection.brand_name,
                'confidence': detection.confidence,
                'region': detection.bbox,
                'type': detection.logo_type
            })
        
        # 2. Pattern recognition
        pattern_signatures = self.pattern_recognizer.detect_patterns(image)
        for signature in pattern_signatures:
            results.append({
                'method': 'pattern_recognition',
                'brand': signature.name,
                'confidence': signature.confidence,
                'region': signature.region,
                'type': signature.signature_type
            })
        
        # 3. Text recognition
        extracted_text = self.text_recognizer.extract_text(image)
        text_signatures = self.text_recognizer.recognize_brands_from_text(extracted_text)
        for signature in text_signatures:
            results.append({
                'method': 'text_recognition',
                'brand': signature.name,
                'confidence': signature.confidence,
                'region': signature.region,
                'type': signature.signature_type
            })
        
        # Aggregate results by brand
        brand_aggregation = self._aggregate_brand_results(results)
        
        return brand_aggregation
    
    def _aggregate_brand_results(self, results: List[Dict]) -> List[Dict]:
        """Aggregate multiple detection results for the same brand"""
        brand_scores = {}
        
        for result in results:
            brand = result['brand']
            confidence = result['confidence']
            method = result['method']
            
            if brand not in brand_scores:
                brand_scores[brand] = {
                    'brand': brand,
                    'confidence': 0,
                    'methods': [],
                    'regions': [],
                    'types': set()
                }
            
            # Weight different methods differently
            method_weights = {
                'logo_detection': 1.0,
                'pattern_recognition': 0.8,
                'text_recognition': 0.6
            }
            
            weighted_confidence = confidence * method_weights.get(method, 0.5)
            brand_scores[brand]['confidence'] += weighted_confidence
            brand_scores[brand]['methods'].append(method)
            brand_scores[brand]['regions'].append(result['region'])
            brand_scores[brand]['types'].add(result['type'])
        
        # Convert to list and sort by confidence
        aggregated_results = []
        for brand, data in brand_scores.items():
            data['confidence'] = min(data['confidence'], 1.0)  # Cap at 1.0
            data['types'] = list(data['types'])
            aggregated_results.append(data)
        
        aggregated_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return aggregated_results
    
    def visualize_detections(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Visualize brand detection results on the image"""
        annotated_image = image.copy()
        
        colors = {
            'logo_detection': (0, 255, 0),    # Green
            'pattern_recognition': (255, 0, 0),  # Blue
            'text_recognition': (0, 0, 255)   # Red
        }
        
        for result in results:
            method = result['method']
            confidence = result['confidence']
            region = result['region']
            
            if method in colors:
                color = colors[method]
                
                # Draw bounding box
                x, y, w, h = region
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{result['brand']}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_image, (x, y - label_size[1] - 10),
                             (x + label_size[0], y), color, -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image

def main():
    """Demo function for brand recognition system"""
    # Initialize brand recognizer
    recognizer = AdvancedBrandRecognizer()
    
    # Create a sample image (in practice, you would load from camera or file)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Perform brand recognition
    results = recognizer.recognize_brands(sample_image)
    
    # Print results
    print("Brand Recognition Results:")
    for result in results:
        print(f"Brand: {result['brand']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Methods: {', '.join(result['methods'])}")
        print(f"Types: {', '.join(result['types'])}")
        print("-" * 40)

if __name__ == "__main__":
    main()
