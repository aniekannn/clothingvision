#!/usr/bin/env python3
"""
Test Improved Detection on Saved Images
Tests the improved detection system on your captured images
"""

import cv2
import logging
from accurate_detection import AccurateFashionDetector
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_on_image(image_path):
    """Test detection on a single image"""
    logger.info(f"Testing detection on: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Create detector
    detector = AccurateFashionDetector()
    
    # Detect clothing
    detections = detector.detect_clothing_on_person(frame)
    
    # Draw results
    result_frame = detector.draw_clothing_detections(frame, detections)
    
    # Add info text
    cv2.putText(result_frame, f"Detected: {len(detections)} items", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Log results
    logger.info(f"Found {len(detections)} clothing items:")
    for detection in detections:
        logger.info(f"  - {detection['category']}: {detection['color']} (confidence: {detection['confidence']:.2f})")
    
    # Display result
    cv2.imshow('Improved Detection Test', result_frame)
    logger.info("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = f"improved_{Path(image_path).name}"
    cv2.imwrite(output_path, result_frame)
    logger.info(f"Result saved to: {output_path}")
    
    return detections

def main():
    """Test on all captured images"""
    # Test images
    test_images = [
        "accurate_detection_capture_20251027_120020.jpg",
        "test_person_image.jpg",
        "real_detection_capture_20251019_212451.jpg"
    ]
    
    results = {}
    
    for image_path in test_images:
        if Path(image_path).exists():
            logger.info(f"\n{'='*60}")
            detections = test_on_image(image_path)
            results[image_path] = detections
        else:
            logger.warning(f"Image not found: {image_path}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY:")
    for image_path, detections in results.items():
        if detections is not None:
            logger.info(f"{image_path}: {len(detections)} items detected")

if __name__ == "__main__":
    main()



