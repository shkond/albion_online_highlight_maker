"""
Video processing utilities for Albion combat clip extraction.
"""
import cv2
import numpy as np


class VideoProcessor:
    """Handles video processing operations."""
    
    # Skill slot positions for 1080p (adjust based on game UI)
    # These are example coordinates - would need calibration for actual game
    SKILL_SLOT_POSITIONS = [
        (760, 960, 64, 64),   # Slot 1 (x, y, width, height)
        (832, 960, 64, 64),   # Slot 2
        (904, 960, 64, 64),   # Slot 3
        (976, 960, 64, 64),   # Slot 4
        (1048, 960, 64, 64),  # Slot 5
        (1120, 960, 64, 64),  # Slot 6
        (1192, 960, 64, 64),  # Slot 7
        (1264, 960, 64, 64),  # Slot 8
    ]
    
    # Screen color check area (center region)
    SCREEN_COLOR_REGION = (540, 380, 200, 200)  # x, y, width, height
    
    # Red screen threshold for death detection
    RED_THRESHOLD = 150
    RED_RATIO_THRESHOLD = 0.6
    # Small epsilon to prevent division by zero in color ratio calculation
    COLOR_EPSILON = 0.1
    
    def __init__(self):
        pass
    
    @staticmethod
    def extract_skill_slots(frame):
        """Extract 8 skill slot images from a frame.
        
        Args:
            frame: numpy array (1080, 1920, 3) BGR format
            
        Returns:
            list: 8 cropped skill slot images
        """
        slots = []
        for x, y, w, h in VideoProcessor.SKILL_SLOT_POSITIONS:
            slot = frame[y:y+h, x:x+w]
            slots.append(slot)
        return slots
    
    @staticmethod
    def is_red_screen(frame):
        """Check if screen shows red (death indicator).
        
        Args:
            frame: numpy array (1080, 1920, 3) BGR format
            
        Returns:
            bool: True if screen is predominantly red
        """
        x, y, w, h = VideoProcessor.SCREEN_COLOR_REGION
        region = frame[y:y+h, x:x+w]
        
        # Calculate average color in BGR
        avg_color = np.mean(region, axis=(0, 1))
        b, g, r = avg_color
        
        # Check if red is dominant and above threshold
        is_red = (r > VideoProcessor.RED_THRESHOLD and 
                  r > b and r > g and 
                  r / (b + g + VideoProcessor.COLOR_EPSILON) > VideoProcessor.RED_RATIO_THRESHOLD)
        
        return is_red
    
    @staticmethod
    def get_video_info(video_path):
        """Get video information.
        
        Args:
            video_path: path to video file
            
        Returns:
            dict: video information (fps, frame_count, duration, width, height)
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
