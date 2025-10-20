"""
Accurate Camera Demo - Only Detects Clothing on Person
Fixes the issue of detecting TVs and faces as clothing
"""

import cv2
import time
import logging
from accurate_detection import AccurateFashionDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccurateCameraDemo:
    """Accurate camera demo that only detects clothing on the person"""
    
    def __init__(self):
        self.detector = AccurateFashionDetector()
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Accurate Camera Demo initialized")
    
    def initialize_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Camera initialized successfully")
        return True
    
    def run_demo(self):
        """Run the accurate detection demo"""
        if not self.initialize_camera():
            return
        
        logger.info("Starting Accurate Fashion Detection Demo...")
        logger.info("Press 'q' to quit, 'c' to capture screenshot")
        logger.info("This version only detects clothing on your body, not background objects!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect clothing items ONLY on the person
                detections = self.detector.detect_clothing_on_person(frame)
                
                # Draw detections
                result_frame = self.detector.draw_clothing_detections(frame, detections)
                
                # Add performance info
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Draw performance info
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_frame, f"Clothing Items: {len(detections)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw instructions
                cv2.putText(result_frame, "Press 'q' to quit, 'c' to capture", (10, result_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Accurate Fashion Detection - Clothing Only", result_frame)
                
                # Log detections every 30 frames
                if self.frame_count % 30 == 0:
                    logger.info(f"Frame {self.frame_count}: Detected {len(detections)} clothing items")
                    for detection in detections:
                        logger.info(f"  - {detection['category']} ({detection['color']}): {detection['confidence']:.2f}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit key pressed")
                    break
                elif key == ord('c'):
                    # Capture screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"accurate_detection_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    logger.info(f"Screenshot saved: {filename}")
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"Demo completed. Processed {self.frame_count} frames")
        logger.info(f"Average FPS: {self.frame_count / elapsed_time:.1f}")

def main():
    """Main function"""
    demo = AccurateCameraDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
