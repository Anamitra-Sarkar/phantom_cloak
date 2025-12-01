"""
PHANTOM-CLOAK: Real-Time Optical Camouflage System
Main Application Loop

A computer vision application that creates an invisibility cloak effect
using MediaPipe for human segmentation and custom VFX for realistic
light-bending distortion effects.
"""

import time
import sys
from typing import Optional
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from vfx_utils import (
    detect_edges,
    apply_predator_shimmer,
    apply_absolute_invisibility,
    refine_mask,
    create_hud_overlay
)


class PhantomCloak:
    """Real-Time Optical Camouflage System."""
    
    CALIBRATION_FRAMES = 30
    CALIBRATION_COUNTDOWN = 3
    FPS_CALCULATION_WINDOW = 30  # Number of frames to use for FPS calculation
    
    MODE_ABSOLUTE = "ABSOLUTE"
    MODE_PREDATOR = "PREDATOR"
    
    # MediaPipe Selfie Segmentation model selection:
    # 0 = General model (slower but works for multiple people)
    # 1 = Landscape model (faster, optimized for single person)
    SEGMENTATION_MODEL = 1
    
    def __init__(self, camera_id: int = 0, target_width: int = 640, target_height: int = 480):
        """
        Initialize the Phantom Cloak system.
        
        Args:
            camera_id: Camera device ID
            target_width: Target frame width for processing
            target_height: Target frame height for processing
        """
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = target_height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.background_plate: Optional[np.ndarray] = None
        self.segmentation = None
        
        self.current_mode = self.MODE_ABSOLUTE
        self.refraction_index = 1.4
        self.is_calibrating = False
        self.calibration_countdown = 0
        
        self.time_offset = 0.0
        self.fps = 0.0
        # Use deque with maxlen for efficient FPS calculation
        self.frame_times: deque = deque(maxlen=self.FPS_CALCULATION_WINDOW)
        
    def initialize_camera(self) -> bool:
        """
        Initialize the camera capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def initialize_segmentation(self) -> bool:
        """
        Initialize MediaPipe Selfie Segmentation.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.segmentation = mp_selfie_segmentation.SelfieSegmentation(
                model_selection=self.SEGMENTATION_MODEL
            )
            return True
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            return False
    
    def calibrate_background(self) -> bool:
        """
        Capture and average background frames for calibration.
        
        Returns:
            True if calibration successful, False otherwise
        """
        if self.cap is None:
            return False
        
        print("Starting background calibration...")
        
        for countdown in range(self.CALIBRATION_COUNTDOWN, 0, -1):
            self.calibration_countdown = countdown
            start_time = time.time()
            
            while time.time() - start_time < 1.0:
                ret, frame = self.cap.read()
                if not ret:
                    return False
                
                frame = cv2.flip(frame, 1)
                display = create_hud_overlay(
                    frame, self.current_mode, self.refraction_index,
                    self.fps, calibrating=True, countdown=countdown
                )
                cv2.imshow("PHANTOM-CLOAK", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
        
        print("Capturing background frames...")
        
        # Use running average to reduce memory usage instead of storing all frames
        running_sum: Optional[np.ndarray] = None
        
        for i in range(self.CALIBRATION_FRAMES):
            ret, frame = self.cap.read()
            if not ret:
                return False
            
            frame = cv2.flip(frame, 1)
            
            # Accumulate frames using running sum for memory efficiency
            if running_sum is None:
                running_sum = frame.astype(np.float64)
            else:
                running_sum += frame.astype(np.float64)
            
            progress = int((i + 1) / self.CALIBRATION_FRAMES * 100)
            display = frame.copy()
            cv2.putText(display, f"Capturing: {progress}%", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("PHANTOM-CLOAK", display)
            cv2.waitKey(1)
        
        # Calculate average from running sum
        self.background_plate = (running_sum / self.CALIBRATION_FRAMES).astype(np.uint8)
        print("Background calibration complete!")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with the invisibility effect.
        
        Args:
            frame: Input camera frame
        
        Returns:
            Processed frame with cloak effect applied
        """
        if self.background_plate is None or self.segmentation is None:
            return frame
        
        height, width = frame.shape[:2]
        bg_height, bg_width = self.background_plate.shape[:2]
        
        if (height, width) != (bg_height, bg_width):
            background = cv2.resize(self.background_plate, (width, height))
        else:
            background = self.background_plate
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.segmentation.process(rgb_frame)
        
        if results.segmentation_mask is None:
            return frame
        
        mask = results.segmentation_mask
        
        refined_mask = refine_mask(mask, blur_size=15)
        
        if self.current_mode == self.MODE_ABSOLUTE:
            result = apply_absolute_invisibility(frame, background, refined_mask)
        else:
            edges = detect_edges(refined_mask)
            result = apply_predator_shimmer(
                frame, background, refined_mask, edges,
                time_offset=self.time_offset,
                refraction_index=self.refraction_index
            )
        
        return result
    
    def update_fps(self) -> None:
        """Update FPS calculation using efficient deque."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Calculate FPS based on time span of frames in deque
        if len(self.frame_times) >= 2:
            time_span = current_time - self.frame_times[0]
            if time_span > 0:
                self.fps = len(self.frame_times) / time_span
            # If time_span is 0 or negative, keep previous FPS value
    
    def toggle_mode(self) -> None:
        """Toggle between invisibility modes."""
        if self.current_mode == self.MODE_ABSOLUTE:
            self.current_mode = self.MODE_PREDATOR
        else:
            self.current_mode = self.MODE_ABSOLUTE
        print(f"Mode switched to: {self.current_mode}")
    
    def run(self) -> None:
        """Main application loop."""
        print("=" * 50)
        print("  PHANTOM-CLOAK: Real-Time Optical Camouflage")
        print("=" * 50)
        
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        if not self.initialize_segmentation():
            print("Failed to initialize segmentation. Exiting.")
            return
        
        cv2.namedWindow("PHANTOM-CLOAK", cv2.WINDOW_NORMAL)
        
        if not self.calibrate_background():
            print("Failed to calibrate background. Exiting.")
            return
        
        print("\nControls:")
        print("  [C] - Recalibrate background")
        print("  [M] - Switch mode (ABSOLUTE/PREDATOR)")
        print("  [+] - Increase refraction index")
        print("  [-] - Decrease refraction index")
        print("  [Q] - Quit")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                self.time_offset = time.time()
                
                processed = self.process_frame(frame)
                
                self.update_fps()
                
                output = create_hud_overlay(
                    processed, self.current_mode, self.refraction_index,
                    self.fps, calibrating=False
                )
                
                cv2.imshow("PHANTOM-CLOAK", output)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("Exiting...")
                    break
                elif key == ord('c') or key == ord('C'):
                    print("Recalibrating...")
                    self.calibrate_background()
                elif key == ord('m') or key == ord('M'):
                    self.toggle_mode()
                elif key == ord('+') or key == ord('='):
                    self.refraction_index = min(2.0, self.refraction_index + 0.1)
                    print(f"Refraction index: {self.refraction_index:.1f}")
                elif key == ord('-') or key == ord('_'):
                    self.refraction_index = max(1.0, self.refraction_index - 0.1)
                    print(f"Refraction index: {self.refraction_index:.1f}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        if self.segmentation is not None:
            self.segmentation.close()
        cv2.destroyAllWindows()
        print("Resources cleaned up.")


def main() -> None:
    """Entry point for the application."""
    camera_id = 0
    
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera ID: {sys.argv[1]}, using default (0)")
    
    app = PhantomCloak(camera_id=camera_id)
    app.run()


if __name__ == "__main__":
    main()
