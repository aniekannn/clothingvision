"""
Fashion Recognition System User Interface
A comprehensive GUI for real-time fashion recognition with camera preview
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
from typing import Dict, List, Optional
import logging
from pathlib import Path

from fashion_recognition_system import FashionRecognitionSystem, DetectionResult
from brand_recognition import AdvancedBrandRecognizer

logger = logging.getLogger(__name__)

class FashionRecognitionUI:
    """Main UI class for the fashion recognition system"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fashion Recognition System")
        self.root.geometry("1200x800")
        
        # Initialize systems
        self.fashion_system = FashionRecognitionSystem()
        self.brand_recognizer = AdvancedBrandRecognizer()
        
        # Camera and processing
        self.camera = None
        self.is_processing = False
        self.current_results = []
        self.processing_thread = None
        
        # UI variables
        self.camera_var = tk.StringVar(value="Camera Off")
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.brand_threshold = tk.DoubleVar(value=0.3)
        self.show_brands = tk.BooleanVar(value=True)
        self.show_attributes = tk.BooleanVar(value=True)
        self.save_results = tk.BooleanVar(value=False)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # History
        self.detection_history = []
        
        self.setup_ui()
        self.setup_camera()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Fashion Recognition System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Center panel - Camera view
        self.setup_camera_panel(main_frame)
        
        # Right panel - Results
        self.setup_results_panel(main_frame)
        
        # Bottom panel - Status and history
        self.setup_status_panel(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera", padding="5")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_button = ttk.Button(camera_frame, text="Start Camera", 
                                       command=self.toggle_camera)
        self.camera_button.pack(fill=tk.X, pady=(0, 5))
        
        camera_status = ttk.Label(camera_frame, textvariable=self.camera_var)
        camera_status.pack(fill=tk.X)
        
        # Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                    variable=self.confidence_threshold, orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X, pady=(0, 5))
        
        confidence_value = ttk.Label(settings_frame, textvariable=self.confidence_threshold)
        confidence_value.pack(anchor=tk.W)
        
        # Brand threshold
        ttk.Label(settings_frame, text="Brand Threshold:").pack(anchor=tk.W)
        brand_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                               variable=self.brand_threshold, orient=tk.HORIZONTAL)
        brand_scale.pack(fill=tk.X, pady=(0, 5))
        
        brand_value = ttk.Label(settings_frame, textvariable=self.brand_threshold)
        brand_value.pack(anchor=tk.W)
        
        # Display options
        display_frame = ttk.LabelFrame(control_frame, text="Display Options", padding="5")
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(display_frame, text="Show Brands", 
                       variable=self.show_brands).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Attributes", 
                       variable=self.show_attributes).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Save Results", 
                       variable=self.save_results).pack(anchor=tk.W)
        
        # Actions
        actions_frame = ttk.LabelFrame(control_frame, text="Actions", padding="5")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(actions_frame, text="Capture Photo", 
                  command=self.capture_photo).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(actions_frame, text="Load Image", 
                  command=self.load_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(actions_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(actions_frame, text="Clear History", 
                  command=self.clear_history).pack(fill=tk.X)
    
    def setup_camera_panel(self, parent):
        """Setup the camera preview panel"""
        camera_frame = ttk.LabelFrame(parent, text="Camera Preview", padding="10")
        camera_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="Camera Preview\nClick 'Start Camera' to begin",
                                     font=("Arial", 12), anchor=tk.CENTER)
        self.camera_label.pack(expand=True, fill=tk.BOTH)
        
        # Performance info
        perf_frame = ttk.Frame(camera_frame)
        perf_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.fps_label = ttk.Label(perf_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT)
        
        self.frame_count_label = ttk.Label(perf_frame, text="Frames: 0")
        self.frame_count_label.pack(side=tk.RIGHT)
    
    def setup_results_panel(self, parent):
        """Setup the results panel"""
        results_frame = ttk.LabelFrame(parent, text="Detection Results", padding="10")
        results_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results display
        self.results_text = tk.Text(results_frame, height=20, width=40, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Results summary
        summary_frame = ttk.Frame(results_frame)
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.summary_label = ttk.Label(summary_frame, text="No detections", font=("Arial", 10, "bold"))
        self.summary_label.pack()
    
    def setup_status_panel(self, parent):
        """Setup the status panel"""
        status_frame = ttk.LabelFrame(parent, text="Status & History", padding="10")
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status
        status_info_frame = ttk.Frame(status_frame)
        status_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_info_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT)
        
        self.processing_label = ttk.Label(status_info_frame, text="")
        self.processing_label.pack(side=tk.RIGHT)
        
        # History listbox
        history_frame = ttk.Frame(status_frame)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(history_frame, text="Detection History:").pack(anchor=tk.W)
        
        self.history_listbox = tk.Listbox(history_frame, height=6)
        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
    
    def setup_camera(self):
        """Setup camera for capture"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Could not open camera")
                self.camera = None
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.camera = None
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_processing:
            self.stop_processing()
        else:
            self.start_processing()
    
    def start_processing(self):
        """Start real-time processing"""
        if not self.camera:
            messagebox.showerror("Error", "Camera not available")
            return
        
        self.is_processing = True
        self.camera_button.config(text="Stop Camera")
        self.camera_var.set("Camera On")
        self.status_label.config(text="Processing...", foreground="blue")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_camera_feed)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Started real-time processing")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        self.camera_button.config(text="Start Camera")
        self.camera_var.set("Camera Off")
        self.status_label.config(text="Stopped", foreground="red")
        
        logger.info("Stopped real-time processing")
    
    def process_camera_feed(self):
        """Process camera feed in separate thread"""
        self.start_time = time.time()
        self.frame_count = 0
        
        while self.is_processing:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Process frame
            start_time = time.time()
            
            # Get fashion recognition results
            fashion_results = self.fashion_system.process_frame(frame)
            
            # Get brand recognition results
            brand_results = self.brand_recognizer.recognize_brands(frame)
            
            processing_time = time.time() - start_time
            
            # Update frame count and FPS
            self.frame_count += 1
            self.fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Combine results
            combined_results = self.combine_results(fashion_results, brand_results)
            
            # Update UI in main thread
            self.root.after(0, self.update_ui, frame, combined_results)
            
            # Save results if enabled
            if self.save_results.get():
                self.save_detection_result(combined_results)
    
    def combine_results(self, fashion_results: List[DetectionResult], 
                       brand_results: List[Dict]) -> List[Dict]:
        """Combine fashion and brand recognition results"""
        combined = []
        
        for result in fashion_results:
            combined_result = {
                'type': 'clothing',
                'category': result.category.value,
                'confidence': result.confidence,
                'bbox': result.bbox,
                'brand': result.brand,
                'brand_confidence': result.brand_confidence,
                'attributes': result.attributes,
                'timestamp': time.time()
            }
            combined.append(combined_result)
        
        for result in brand_results:
            combined_result = {
                'type': 'brand',
                'brand': result['brand'],
                'confidence': result['confidence'],
                'methods': result['methods'],
                'types': result['types'],
                'timestamp': time.time()
            }
            combined.append(combined_result)
        
        return combined
    
    def update_ui(self, frame: np.ndarray, results: List[Dict]):
        """Update UI with new results"""
        # Update camera display
        self.update_camera_display(frame, results)
        
        # Update results text
        self.update_results_display(results)
        
        # Update performance info
        self.update_performance_info()
        
        # Store current results
        self.current_results = results
    
    def update_camera_display(self, frame: np.ndarray, results: List[Dict]):
        """Update camera display with annotations"""
        # Draw annotations on frame
        annotated_frame = self.draw_annotations(frame, results)
        
        # Convert to PIL Image
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize to fit display
        display_size = (640, 480)
        pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo  # Keep a reference
    
    def draw_annotations(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw annotations on the frame"""
        annotated_frame = frame.copy()
        
        for result in results:
            if result['type'] == 'clothing':
                x, y, w, h = result['bbox']
                confidence = result['confidence']
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > self.confidence_threshold.get() else (0, 255, 255)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                
                # Prepare label
                label_parts = [f"{result['category']}: {confidence:.2f}"]
                
                if self.show_brands.get() and result['brand']:
                    label_parts.append(f"Brand: {result['brand']}")
                
                label = " | ".join(label_parts)
                
                # Draw label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10),
                             (x + label_size[0], y), color, -1)
                cv2.putText(annotated_frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def update_results_display(self, results: List[Dict]):
        """Update the results text display"""
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No detections found")
            self.summary_label.config(text="No detections")
            return
        
        # Write results
        for i, result in enumerate(results):
            self.results_text.insert(tk.END, f"Detection {i+1}:\n")
            
            if result['type'] == 'clothing':
                self.results_text.insert(tk.END, f"  Category: {result['category']}\n")
                self.results_text.insert(tk.END, f"  Confidence: {result['confidence']:.3f}\n")
                
                if result['brand']:
                    self.results_text.insert(tk.END, f"  Brand: {result['brand']}\n")
                    if result['brand_confidence']:
                        self.results_text.insert(tk.END, f"  Brand Confidence: {result['brand_confidence']:.3f}\n")
                
                if self.show_attributes.get() and result['attributes']:
                    self.results_text.insert(tk.END, f"  Attributes: {', '.join(result['attributes'].keys())}\n")
                
                self.results_text.insert(tk.END, f"  Location: {result['bbox']}\n")
            
            elif result['type'] == 'brand':
                self.results_text.insert(tk.END, f"  Brand: {result['brand']}\n")
                self.results_text.insert(tk.END, f"  Confidence: {result['confidence']:.3f}\n")
                self.results_text.insert(tk.END, f"  Methods: {', '.join(result['methods'])}\n")
                self.results_text.insert(tk.END, f"  Types: {', '.join(result['types'])}\n")
            
            self.results_text.insert(tk.END, "\n")
        
        # Update summary
        clothing_count = len([r for r in results if r['type'] == 'clothing'])
        brand_count = len([r for r in results if r['type'] == 'brand'])
        self.summary_label.config(text=f"{clothing_count} clothing items, {brand_count} brands detected")
    
    def update_performance_info(self):
        """Update performance information"""
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.frame_count_label.config(text=f"Frames: {self.frame_count}")
        
        # Update processing indicator
        if self.is_processing:
            self.processing_label.config(text="● Processing")
        else:
            self.processing_label.config(text="")
    
    def capture_photo(self):
        """Capture a photo from the camera"""
        if not self.camera or not self.is_processing:
            messagebox.showwarning("Warning", "Camera not running")
            return
        
        ret, frame = self.camera.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture photo")
            return
        
        # Save photo
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Success", f"Photo saved to {filename}")
    
    def load_image(self):
        """Load and process an image file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Load image
                image = cv2.imread(filename)
                if image is None:
                    messagebox.showerror("Error", "Could not load image")
                    return
                
                # Process image
                fashion_results = self.fashion_system.process_frame(image)
                brand_results = self.brand_recognizer.recognize_brands(image)
                combined_results = self.combine_results(fashion_results, brand_results)
                
                # Update UI
                self.update_camera_display(image, combined_results)
                self.update_results_display(combined_results)
                self.current_results = combined_results
                
                # Add to history
                self.add_to_history(f"Loaded: {Path(filename).name}", combined_results)
                
                messagebox.showinfo("Success", "Image processed successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {e}")
    
    def export_results(self):
        """Export detection results to file"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(self.current_results, f, indent=2, default=str)
                else:
                    with open(filename, 'w') as f:
                        for result in self.current_results:
                            f.write(f"{result}\n")
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results: {e}")
    
    def clear_history(self):
        """Clear detection history"""
        self.history_listbox.delete(0, tk.END)
        self.detection_history.clear()
    
    def add_to_history(self, description: str, results: List[Dict]):
        """Add detection result to history"""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"{timestamp} - {description}"
        
        self.history_listbox.insert(0, entry)
        self.detection_history.insert(0, {
            'description': description,
            'timestamp': timestamp,
            'results': results
        })
        
        # Limit history size
        if len(self.detection_history) > 100:
            self.detection_history.pop()
            self.history_listbox.delete(tk.END)
    
    def on_history_select(self, event):
        """Handle history selection"""
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.detection_history):
                history_item = self.detection_history[index]
                self.update_results_display(history_item['results'])
    
    def save_detection_result(self, results: List[Dict]):
        """Save detection result to history"""
        if results:
            self.add_to_history("Real-time detection", results)
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.is_processing:
            self.stop_processing()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        logger.info("Application cleanup completed")

def main():
    """Main function to run the fashion recognition UI"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the UI
    app = FashionRecognitionUI()
    app.run()

if __name__ == "__main__":
    main()
